"""Read-out after training: noise ceiling, bootstrap intervals and context sweep.

None of these feed checkpoint selection; they run once per saved checkpoint.
"""

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from core.precision_recall import precision_recall
from core.surrogate_cnp import split_context_target
from core.surrogates.base import Episode
from core.surrogates.evaluation import evaluate_surrogate

# Resamples for the noise ceiling and the bootstrap intervals.
READOUT_DRAWS = 200
# SeedSequence keys under validation_seed (0 is the selection voxel subset).
CEILING_KEY, BOOTSTRAP_KEY = 1, 2
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
    """95% intervals over voxels resampled with replacement: log loss, AP, Spearman r."""
    rng = _rng(seed, BOOTSTRAP_KEY)
    mask, n = target.event_mask, target.events_per_voxel  # [B, N], [B]
    loss = np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits))
    voxel_loss = np.where(mask, loss, 0.0).sum(axis=1)  # [B]
    observed = np.where(mask, labels, 0).sum(axis=1) / n  # [B]
    predicted = np.where(mask, expit(logits), 0.0).sum(axis=1) / n  # [B]
    samples = {"bernoulli_log_loss": [], "average_precision": [], "spearman_r": []}
    for _ in range(draws):
        index = rng.integers(0, len(n), len(n))
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


def readout_metrics(target, arrays, *, seed):
    """Noise ceiling and bootstrap intervals for one evaluated episode."""
    probabilities = expit(arrays["logits"])
    return dict(
        spearman_noise_ceiling=spearman_noise_ceiling(target, probabilities, seed=seed),
        **bootstrap_intervals(target, arrays["labels"], arrays["logits"], seed=seed),
    )


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
