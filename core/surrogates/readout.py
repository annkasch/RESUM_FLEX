"""Read-out after training: noise ceiling, bootstrap intervals, within-design-point null,
feature calibration and context sweep.

None of these feed checkpoint selection; they run once per saved checkpoint.
"""

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from core.precision_recall import precision_recall
from core.surrogate_cnp import split_context_target
from core.surrogates.base import Episode
from core.surrogates.evaluation import (
    evaluate_surrogate,
    feature_calibration,
    feature_columns,
    within_point_summary,
    within_point_terms,
)

# Resamples for the noise ceiling and the bootstrap intervals.
READOUT_DRAWS = 200
# Label draws for the within-design-point null; enough for its 3σ quantiles.
NULL_DRAWS = 1000
# Two-sided Gaussian tail probabilities of the ±1/2/3σ null quantiles.
NULL_TAILS = {1: 0.158655, 2: 0.0227501, 3: 0.0013499}
# SeedSequence keys under validation_seed (0 is the selection design-point subset).
CEILING_KEY, BOOTSTRAP_KEY, NULL_KEY = 1, 2, 3
SWEEP_METRICS = (
    "bernoulli_log_loss",
    "log_loss_skill",
    "average_precision",
    "lift",
    "spearman_r",
    "voxel_rate_mse_denoised",
    "calibration_z",
    "voxel_z_mean",
    "voxel_z2_mean",
)


def _rng(seed, key):
    return np.random.default_rng(np.random.SeedSequence([seed, key]))


def _spearman(observed, predicted):
    if np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        return np.nan
    return float(spearmanr(observed, predicted).statistic)


def spearman_noise_ceiling(target, probabilities, *, seed, draws=READOUT_DRAWS):
    """Median Spearman r a model equal to the truth would reach at this counting noise.

    Labels are redrawn from the model's own probabilities [B, N] and correlated with
    its predicted voxel rates.
    """
    rng = _rng(seed, CEILING_KEY)
    mask, n = target.event_mask, target.events_per_voxel  # [B, N], [B]
    predicted = np.where(mask, probabilities, 0.0).sum(axis=1) / n  # [B]
    values = []
    for _ in range(draws):
        hits = ((rng.random(probabilities.shape) < probabilities) & mask).sum(axis=1)  # [B]
        values.append(_spearman(hits / n, predicted))
    values = np.asarray(values)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else None


def bootstrap_intervals(target, labels, logits, *, seed, draws=READOUT_DRAWS):
    """95% intervals over design points resampled with replacement.

    Log loss, AP, Spearman r and the within-design-point log-loss gain and AUC.
    """
    rng = _rng(seed, BOOTSTRAP_KEY)
    mask, n = target.event_mask, target.events_per_voxel  # [B, N], [B]
    loss = np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits))
    voxel_loss = np.where(mask, loss, 0.0).sum(axis=1)  # [B]
    observed = np.where(mask, labels, 0).sum(axis=1) / n  # [B]
    predicted = np.where(mask, expit(logits), 0.0).sum(axis=1) / n  # [B]
    terms = within_point_terms(target, labels, logits)
    within = ("within_point_log_loss_gain", "within_point_auc")
    samples = {"bernoulli_log_loss": [], "average_precision": [], "spearman_r": []}
    samples.update({key: [] for key in within})
    for _ in range(draws):
        index = rng.integers(0, len(n), len(n))
        resampled = within_point_summary({k: v[index] for k, v in terms.items() if v.ndim == 1})
        for key in within:
            samples[key].append(np.nan if resampled[key] is None else resampled[key])
        samples["bernoulli_log_loss"].append(voxel_loss[index].sum() / n[index].sum())
        y, s = labels[index][mask[index]], logits[index][mask[index]]
        samples["average_precision"].append(
            precision_recall(y, s)["average_precision"] if y.any() else np.nan
        )
        samples["spearman_r"].append(_spearman(observed[index], predicted[index]))
    intervals = {}
    for key, values in samples.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        intervals[f"{key}_ci95"] = (
            [float(v) for v in np.percentile(finite, [2.5, 97.5])] if finite.size else None
        )
    return intervals


def within_point_null(target, logits, *, seed, draws=NULL_DRAWS):
    """Quantiles of each design point's log-loss gain if labels followed the model's p.

    Labels are redrawn from the model's own probabilities; the gain a·y + c of each draw
    gives the null distribution per design point. Returns {k: (lower, upper)} [B] at the
    two-sided Gaussian ±kσ tail probabilities, k = 1, 2, 3.
    """
    rng = _rng(seed, NULL_KEY)
    labels = np.zeros(target.labels.shape)  # labels do not enter a, c, p
    terms = within_point_terms(target, labels, logits)
    a, c, p = terms["a"], terms["c"], terms["p"]  # [B, N]
    offset = c.sum(axis=1)  # [B]
    gains = np.empty((draws, len(offset)))  # [draws, B]
    for d in range(draws):
        gains[d] = (a * (rng.random(p.shape) < p)).sum(axis=1) + offset
    return {
        k: (np.quantile(gains, tail, axis=0), np.quantile(gains, 1 - tail, axis=0))
        for k, tail in NULL_TAILS.items()
    }


def readout_metrics(target, arrays, *, seed, residual=None):
    """Noise ceiling, bootstrap intervals, within-design-point null coverage, feature
    calibration and, with ``residual`` (ResidualTestConfig), the residual learnability
    test for one evaluated episode. Returns (metrics, arrays)."""
    probabilities = expit(arrays["logits"])
    metrics = dict(
        spearman_noise_ceiling=spearman_noise_ceiling(target, probabilities, seed=seed),
        **bootstrap_intervals(target, arrays["labels"], arrays["logits"], seed=seed),
    )
    null = within_point_null(target, arrays["logits"], seed=seed)
    # Design points with constant predictions have a degenerate null; they are skipped.
    spread = arrays["within_point_gain_variance"] > 0  # [B]
    gain = arrays["within_point_gain"][spread]
    metrics["within_point_gain_coverage"] = (
        {
            f"{k}sigma": float(((gain >= lower[spread]) & (gain <= upper[spread])).mean())
            for k, (lower, upper) in null.items()
        }
        if spread.any()
        else None
    )
    extra = {"within_point_gain_null": np.stack([np.stack(null[k]) for k in (1, 2, 3)])}
    mask = target.event_mask  # [B, N]
    names, values = feature_columns(target)
    if names:
        metrics["feature_calibration"], feature_arrays = feature_calibration(
            names, values[mask], arrays["labels"][mask], probabilities[mask]
        )
        extra.update(feature_arrays)
    if residual is not None:
        from core.surrogates.residual import residual_test

        rows = np.nonzero(mask)[0]  # [E] design point of each valid event
        metrics["residual_test"], residual_arrays = residual_test(
            values[mask], arrays["labels"][mask], arrays["logits"][mask], rows, residual, seed=seed
        )
        if residual_arrays:
            for key in ("residual_logits", "residual_fold"):
                full = np.full(mask.shape, np.nan if key == "residual_logits" else -1)
                full[mask] = residual_arrays.pop(key)
                extra[key] = full  # [B, N]; padding NaN / -1
            extra.update(residual_arrays)
    return metrics, extra


def context_sweep(model, batch, config, *, reference_rate=None):
    """Metrics at each context size on one fixed target set, plus zeroed context labels.

    All sizes share the seeded layout of ``config.validation_context_pool``. The
    zeroed run uses the standard size with every context label set to 0: if it
    scores like the observed labels, the model ignores them.
    """
    sizes = sorted({config.validation_context_events, *config.validation_context_sweep})
    rows = []
    for n_context in sizes:
        context, target = split_context_target(
            batch,
            n_context,
            seed=config.validation_seed,
            n_pool=config.validation_context_pool,
        )
        variants = [("observed", context)]
        if n_context == config.validation_context_events:
            zeroed = context.model_copy(update={"labels": np.zeros_like(context.labels)})
            variants.append(("zeroed", zeroed))
        for labels, episode_context in variants:
            episode = Episode(episode_context, target)
            metrics, _ = evaluate_surrogate(
                model, episode.target, context=episode.context, reference_rate=reference_rate
            )
            rows.append(
                dict(
                    n_context=n_context,
                    context_labels=labels,
                    **{key: metrics[key] for key in SWEEP_METRICS},
                )
            )
    return rows
