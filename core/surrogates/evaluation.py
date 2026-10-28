"""One set of metrics and output arrays for all event models."""

from contextlib import contextmanager

import numpy as np
from scipy.stats import spearmanr

from core.precision_recall import precision_recall

# Equal-count bins of predicted probability for the calibration curve.
CALIBRATION_BINS = 10

_EVALUATION_THREADS = None


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


def calibration_bins(labels, probabilities, n_bins=CALIBRATION_BINS):
    """Equal-count bins of predicted p: events, hits, Σp and Σp(1-p) per bin.

    labels, probabilities: [E] valid events only.
    """
    order = np.argsort(probabilities, kind="stable")
    bins = np.array_split(order, min(n_bins, len(order)))
    p, y = probabilities, labels
    return dict(
        calibration_events=np.array([len(b) for b in bins]),
        calibration_hits=np.array([y[b].sum() for b in bins], dtype=float),
        calibration_predicted=np.array([p[b].sum() for b in bins]),
        calibration_variance=np.array([(p[b] * (1 - p[b])).sum() for b in bins]),
    )


def probability_metrics(target, labels, probabilities, *, reference_rate=None):
    """Calibration and voxel-trend metrics of event probabilities against natural labels.

    target: StandardBatch giving the voxel layout; labels, probabilities: [B, N].
    reference_rate: constant-rate model for the log-loss skill (training prevalence).
    """
    mask = target.event_mask                                       # [B, N]
    n = target.events_per_voxel                                    # [B]
    if np.any(n < 2):
        raise ValueError("Voxel metrics need at least two target events per voxel")
    p = np.where(mask, probabilities, 0.0)                         # [B, N]
    hits = np.where(mask, labels, 0).sum(axis=1).astype(float)     # [B]
    expected = p.sum(axis=1)                                       # [B]
    variance = (p * (1 - p)).sum(axis=1)                           # [B] binomial variance of hits
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
    chi2 = np.square(bins["calibration_hits"] - bins["calibration_predicted"]) / bins[
        "calibration_variance"
    ]
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
    )
    if prediction.legacy_scale is not None:
        arrays["legacy_scale"] = prediction.legacy_scale
    if target.event_counts is not None:
        arrays["event_counts"] = target.event_counts
    return metrics, arrays
