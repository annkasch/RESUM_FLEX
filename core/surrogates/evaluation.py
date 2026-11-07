"""One set of metrics and output arrays for all event models."""

from contextlib import contextmanager

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

from core.precision_recall import precision_recall

# Equal-count bins of predicted probability (calibration curve) or of one input feature.
CALIBRATION_BINS = 10

# |logit p_i - logit p̄_b| below this is rounding of a constant prediction, not spread.
SLOPE_ROUNDING = 1e-9

_EVALUATION_THREADS = None

# Central bootstrap intervals as percentiles: ±1σ (68 %) and 95 %, both from the same draws.
INTERVAL_PERCENTILES = {"ci68": (15.865525, 84.134475), "ci95": (2.5, 97.5)}


def bootstrap_bands(key, values):
    """{key_ci68, key_ci95}: central intervals of finite bootstrap values (None if none)."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        f"{key}_{name}": [float(v) for v in np.percentile(finite, q)] if finite.size else None
        for name, q in INTERVAL_PERCENTILES.items()
    }


@contextmanager
def evaluation_threads(n: int):
    """Run every evaluate_surrogate call inside the block with ``n`` torch CPU threads."""
    global _EVALUATION_THREADS
    previous, _EVALUATION_THREADS = _EVALUATION_THREADS, n
    try:
        yield
    finally:
        _EVALUATION_THREADS = previous


def _predict(model, target, context):
    if _EVALUATION_THREADS is None:
        return model.predict(target, context=context)
    import torch

    training_threads = torch.get_num_threads()
    torch.set_num_threads(_EVALUATION_THREADS)
    try:
        return model.predict(target, context=context)
    finally:
        torch.set_num_threads(training_threads)


def calibration_bins(labels, probabilities, n_bins=CALIBRATION_BINS, *, sort_by=None):
    """Equal-count bins of predicted p (or of ``sort_by``): events, hits, Σp, Σp(1-p).

    labels, probabilities, sort_by: [E] valid events only.
    """
    key = probabilities if sort_by is None else sort_by
    order = np.argsort(key, kind="stable")
    bins = np.array_split(order, min(n_bins, len(order)))
    p, y = probabilities, labels
    return dict(
        calibration_events=np.array([len(b) for b in bins]),
        calibration_hits=np.array([y[b].sum() for b in bins], dtype=float),
        calibration_predicted=np.array([p[b].sum() for b in bins]),
        calibration_variance=np.array([(p[b] * (1 - p[b])).sum() for b in bins]),
    )


def feature_columns(target):
    """Per-event input columns of a design-point batch: (names, values [B, N, F]).

    θ columns are repeated over the events of their design point; φ columns follow.
    """
    shape = target.labels.shape  # [B, N]
    names, columns = [], []
    if target.theta is not None:
        theta = np.asarray(target.theta, dtype=float)  # [B, D_θ]
        names += [f"theta_{j}" for j in range(theta.shape[1])]
        columns += [np.broadcast_to(theta[:, None, j], shape) for j in range(theta.shape[1])]
    if target.phi is not None:
        phi = np.asarray(target.phi, dtype=float)  # [B, N, D_φ]
        names += [f"phi_{j}" for j in range(phi.shape[2])]
        columns += [phi[..., j] for j in range(phi.shape[2])]
    values = np.stack(columns, axis=-1) if columns else np.zeros((*shape, 0))  # [B, N, F]
    return names, values


def feature_calibration(names, features, labels, probabilities, n_bins=CALIBRATION_BINS):
    """Calibration along each input feature, pooled over all events and design points.

    names: F column names; features: [E, F]; labels, probabilities: [E] valid events.
    Per feature, equal-count bins of its value: z = (hits - Σp) / sqrt(Σp(1-p)) per bin.
    Returns (metrics per feature, arrays [F, bins] incl. the median feature value per bin).
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] != len(labels) or len(names) != features.shape[1]:
        raise ValueError("features must be [E, F] with one name per column")
    if not np.isfinite(features).all():
        raise ValueError("Feature values must be finite for feature calibration")
    n_bins = min(n_bins, len(labels))
    keys = ("events", "hits", "predicted", "variance", "value")
    arrays = {f"feature_calibration_{k}": np.zeros((len(names), n_bins)) for k in keys}
    metrics = {}
    for f, name in enumerate(names):
        bins = calibration_bins(labels, probabilities, n_bins, sort_by=features[:, f])
        order = np.argsort(features[:, f], kind="stable")
        median = [np.median(features[b, f]) for b in np.array_split(order, n_bins)]
        for k in keys[:4]:
            arrays[f"feature_calibration_{k}"][f] = bins[f"calibration_{k}"]
        arrays["feature_calibration_value"][f] = median
        variance = bins["calibration_variance"]
        if np.any(variance <= 0):
            raise ValueError(f"Predicted hit variance is zero in a bin of {name}")
        z = (bins["calibration_hits"] - bins["calibration_predicted"]) / np.sqrt(variance)
        metrics[name] = dict(
            chi2_per_bin=float(np.square(z).mean()), max_abs_z=float(np.abs(z).max())
        )
    arrays["feature_calibration_names"] = np.asarray(names)
    return metrics, arrays


def within_point_terms(target, labels, logits):
    """Per-design-point terms of the within-design-point log-loss gain and pair AUC.

    Baseline: every event of design point b gets the model's own mean p̄_b. The gain of
    event i is loss(p̄_b) - loss(p_i) = a_i y_i + c_i with a_i = logit p_i - logit p̄_b and
    c_i = log(1-p_i) - log(1-p̄_b). If labels followed the model's p, its mean would be
    KL(p_i || p̄_b) = a_i p_i + c_i and its variance p_i(1-p_i) a_i².
    target: StandardBatch; labels, logits: [B, N]. Returns arrays [B] and a, c, p [B, N].
    """
    mask = target.event_mask  # [B, N]
    n = target.events_per_voxel  # [B]
    logits = np.asarray(logits, dtype=float)
    p = np.where(mask, expit(logits), 0.0)  # [B, N]
    mean = p.sum(axis=1) / n  # [B]
    if np.any(mean <= 0) or np.any(mean >= 1):
        raise ValueError("Design-point mean probabilities must lie strictly inside (0, 1)")
    log_q = -np.logaddexp(0, logits)  # [B, N] log(1 - p)
    a = np.where(mask, logits - np.log(mean / (1 - mean))[:, None], 0.0)  # [B, N]
    a = np.where(np.abs(a) < SLOPE_ROUNDING, 0.0, a)  # equal logits: rounding only
    c = np.where(mask, log_q - np.log1p(-mean)[:, None], 0.0)  # [B, N]
    y = np.where(mask, labels, 0).astype(float)  # [B, N]
    hits = y.sum(axis=1)  # [B]
    # Mann-Whitney within each design point; padding ranks above every valid event.
    ranks = rankdata(np.where(mask, logits, np.inf), axis=1)  # [B, N], ties averaged
    loss = np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits))
    return dict(
        gain=(a * y + c).sum(axis=1),
        expected=(a * p + c).sum(axis=1),
        variance=(p * (1 - p) * np.square(a)).sum(axis=1),
        hits=hits,
        model_loss=np.where(mask, loss, 0.0).sum(axis=1),
        auc_u=(ranks * y).sum(axis=1) - hits * (hits + 1) / 2,
        auc_pairs=hits * (n - hits),
        a=a,
        c=c,
        p=p,
    )


def within_point_summary(terms):
    """Pooled within-design-point metrics from per-design-point terms (all [B])."""
    gain, hits, pairs = terms["gain"].sum(), terms["hits"].sum(), terms["auc_pairs"].sum()
    variance = terms["variance"].sum()
    return dict(
        within_point_log_loss_gain=float(gain / hits) if hits else None,
        within_point_log_loss_skill=float(gain / (terms["model_loss"].sum() + gain)),
        within_point_auc=float(terms["auc_u"].sum() / pairs) if pairs else None,
        within_point_spread_z=float((gain - terms["expected"].sum()) / np.sqrt(variance))
        if variance > 0
        else None,
    )


def probability_metrics(target, labels, probabilities, *, reference_rate=None):
    """Calibration and voxel-trend metrics of event probabilities against natural labels.

    target: StandardBatch giving the voxel layout; labels, probabilities: [B, N].
    reference_rate: constant-rate model for the log-loss skill (training prevalence).
    """
    mask = target.event_mask  # [B, N]
    n = target.events_per_voxel  # [B]
    if np.any(n < 2):
        raise ValueError("Voxel metrics need at least two target events per voxel")
    p = np.where(mask, probabilities, 0.0)  # [B, N]
    hits = np.where(mask, labels, 0).sum(axis=1).astype(float)  # [B]
    expected = p.sum(axis=1)  # [B]
    variance = (p * (1 - p)).sum(axis=1)  # [B] binomial variance of hits
    if np.any(variance <= 0):
        raise ValueError("Predicted hit variance is zero; probabilities are degenerate")
    observed, predicted = hits / n, expected / n
    # (o - q)^2 minus the unbiased binomial noise estimate o(1-o)/(n-1): on average (t - q)^2.
    denoised = np.square(observed - predicted) - observed * (1 - observed) / (n - 1)
    z = (hits - expected) / np.sqrt(variance)
    spearman = (
        float(spearmanr(observed, predicted).statistic)
        if np.ptp(observed) > 0 and np.ptp(predicted) > 0
        else None
    )
    bins = calibration_bins(labels[mask], probabilities[mask])
    chi2 = (
        np.square(bins["calibration_hits"] - bins["calibration_predicted"])
        / bins["calibration_variance"]
    )
    metrics = dict(
        voxel_rate_mse_denoised=float(denoised.mean()),
        spearman_r=spearman,
        calibration_z=float((hits.sum() - expected.sum()) / np.sqrt(variance.sum())),
        voxel_z_mean=float(z.mean()),
        voxel_z2_mean=float(np.square(z).mean()),
        calibration_chi2_per_bin=float(chi2.mean()),
        log_loss_skill=None,
    )
    if reference_rate is not None:
        if not 0 < reference_rate < 1:
            raise ValueError("reference_rate must lie in (0, 1)")
        y = labels[mask]
        reference = -np.where(y == 1, np.log(reference_rate), np.log1p(-reference_rate)).mean()
        metrics["reference_log_loss"] = float(reference)
    return metrics, dict(voxel_z=z, **bins)


def _count_nll(hits, trials, probability):
    """Mean -log Binomial(k | n, p) over voxels with the predicted mean p (plug-in)."""
    from scipy.stats import binom

    if np.any(probability <= 0) or np.any(probability >= 1):
        raise ValueError("Voxel mean probabilities must lie strictly inside (0, 1)")
    return float(-binom.logpmf(hits, trials, probability).mean())


def evaluate_surrogate(model, target, *, context=None, reference_rate=None):
    prediction = _predict(model, target, context)
    labels = np.asarray(target.labels)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Evaluation requires real binary labels, not mixup targets")
    logits = prediction.logits
    if logits.shape != labels.shape:
        raise ValueError("Predictions and labels must have identical shapes")
    observed, predicted = target.voxel_mean(labels), prediction.mean
    residual = predicted - observed
    # Event-level metrics use only valid events; padded positions are dropped.
    mask = target.event_mask
    event_labels, event_logits = labels[mask], logits[mask]
    curve = (
        precision_recall(event_labels, event_logits)
        if labels.sum()
        else dict(
            average_precision=None,
            prevalence=0.0,
            precision=np.array([]),
            recall=np.array([]),
            logit_threshold=np.array([]),
        )
    )
    mean_observed = float(observed.mean())
    metrics = dict(
        voxels=len(observed),
        events=int(event_labels.size),
        positives=int(event_labels.sum()),
        mean_observed=mean_observed,
        mean_predicted=float(predicted.mean()),
        mean_bias=float(residual.mean()),
        ratio_of_means=float(predicted.mean() / mean_observed) if mean_observed else None,
        pearson_r=float(np.corrcoef(observed, predicted)[0, 1])
        if observed.std() > 0 and predicted.std() > 0
        else None,
        voxel_rate_mae=float(np.abs(residual).mean()),
        voxel_rate_rmse=float(np.sqrt(np.square(residual).mean())),
        bernoulli_log_loss=float(
            np.where(
                event_labels == 1, np.logaddexp(0, -event_logits), np.logaddexp(0, event_logits)
            ).mean()
        ),
        brier_score=float(np.square(prediction.probabilities[mask] - event_labels).mean()),
        voxel_count_nll=_count_nll(
            np.where(mask, labels, 0).sum(axis=1), target.events_per_voxel, predicted
        ),
        average_precision=curve["average_precision"],
        prevalence=curve["prevalence"],
        lift=curve["average_precision"] / curve["prevalence"] if labels.sum() else None,
    )
    extra, extra_arrays = probability_metrics(
        target, labels, prediction.probabilities, reference_rate=reference_rate
    )
    reference = extra.pop("reference_log_loss", None)
    if reference is not None:
        extra["log_loss_skill"] = 1 - metrics["bernoulli_log_loss"] / reference
    metrics.update(extra)
    terms = within_point_terms(target, labels, logits)
    metrics.update(within_point_summary(terms))
    arrays = dict(
        logits=logits,
        labels=labels,
        observed=observed,
        predicted=predicted,
        residual=residual,
        precision=curve["precision"],
        recall=curve["recall"],
        logit_threshold=curve["logit_threshold"],
        **extra_arrays,
        within_point_gain=terms["gain"],
        within_point_expected_gain=terms["expected"],
        within_point_gain_variance=terms["variance"],
    )
    if prediction.legacy_scale is not None:
        arrays["legacy_scale"] = prediction.legacy_scale
    if target.event_counts is not None:
        arrays["event_counts"] = target.event_counts
    return metrics, arrays
