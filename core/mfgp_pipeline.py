"""Glue between event models and the multi-fidelity GP, for any number of levels.

This module is the only place where torch and GPy meet, and only at the numpy boundary:
event models turn batches into per-voxel arrays, which :func:`fit_mfgp` fits as GP
levels ordered from the lowest to the highest fidelity.

Each level is one (X, Y) pair built from a batch:

* ``model_mean``: mean predicted p over each voxel's target events (an
  :class:`EventSurrogate` or the historical two-output :class:`ConditionalNeuralProcess`).
* ``observed_fraction``: hit fraction m/N over the same target events.

With ``n_context`` each voxel's events are split into context and targets with ``seed``;
levels built from the same batch, ``n_context`` and ``seed`` share their targets.

EVENT_ONLY data have no ``θ`` and can't drive a GP; the entry points reject them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.surrogate_cnp import ConditionalNeuralProcess, split_context_target
from core.surrogate_mfgp import MultiFidelityGP
from core.surrogates.base import EventSurrogate
from core.training import cnp_trial_predictive
from data.pseudo_generator import PseudoDataGenerator
from schemas.data_models import InputMode, StandardBatch

SOURCES = ("model_mean", "observed_fraction")


def model_mean(
    model: EventSurrogate | ConditionalNeuralProcess,
    batch: StandardBatch,
    *,
    n_context: int | None = None,
    seed: int = 0,
    n_mc_samples: int = 50,
) -> np.ndarray:
    """Mean predicted p per voxel [B] over the target events.

    Context models need ``n_context``; context-free models use every event unless
    ``n_context`` is given (then the same targets as an observed fraction).
    ``n_mc_samples`` applies only to the two-output CNP.
    """
    legacy = isinstance(model, ConditionalNeuralProcess)
    uses_context = legacy or model.uses_context
    if n_context is None:
        if uses_context:
            raise ValueError("Context models need n_context for their context/target split")
        return model.predict(batch).mean
    context, target = split_context_target(batch, n_context, seed=seed)
    if legacy:
        return cnp_trial_predictive(model, context, target, n_mc_samples=n_mc_samples)["y_cnp"]
    return model.predict(target, context=context if uses_context else None).mean


def observed_fraction(
    batch: StandardBatch, *, n_context: int | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Hit fraction [B] and event count [B] per voxel, over the target events if split."""
    if n_context is not None:
        batch = split_context_target(batch, n_context, seed=seed)[1]
    hits = np.where(batch.event_mask, batch.labels, 0).sum(axis=1).astype(float)
    events = batch.events_per_voxel.astype(float)
    return hits / events, events


@dataclass(frozen=True)
class BatchLevel:
    """One GP level built from a batch; see the module docstring."""

    source: str
    batch: StandardBatch
    n_context: int | None = None
    seed: int = 0

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        _reject_event_only(self.batch, f"{self.source} batch")


def levels_from_batches(
    model: EventSurrogate | ConditionalNeuralProcess | None,
    levels: list[BatchLevel],
    *,
    n_mc_samples: int = 50,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """X_list [n_k, D_θ] and Y_list [n_k, 1], lowest fidelity first; pass to fit_mfgp."""
    X_list, Y_list = [], []
    for level in levels:
        if level.source == "model_mean":
            if model is None:
                raise ValueError("model_mean levels need an event model")
            y = model_mean(
                model,
                level.batch,
                n_context=level.n_context,
                seed=level.seed,
                n_mc_samples=n_mc_samples,
            )
        else:
            y = observed_fraction(level.batch, n_context=level.n_context, seed=level.seed)[0]
        X_list.append(level.batch.theta)
        Y_list.append(np.asarray(y, dtype=float).reshape(-1, 1))
    return X_list, Y_list


def fit_targets(
    values: np.ndarray,
    events: np.ndarray,
    *,
    source: str,
    output_transform: str,
    pseudocount: float = 0.5,
    floor: float = 1e-12,
) -> np.ndarray:
    """GP target [n] of one level in original units, from values [n] and events [n].

    Identity mode: unchanged. Log mode: observed fractions become (m + a)/(N + 2a)
    with a = pseudocount; model means must be finite and nonnegative and are floored.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    values = np.asarray(values, dtype=float)
    if output_transform != "log":
        return values
    if source == "observed_fraction":
        events = np.asarray(events, dtype=float)
        return (values * events + pseudocount) / (events + 2 * pseudocount)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Log-space MFGP requires nonnegative finite model means")
    return np.maximum(values, floor)


def binomial_log_moments(
    probability: np.ndarray, events: np.ndarray, pseudocount: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and variance [n] of log((m + a)/(N + 2a)) for m ~ Binomial(N, p).

    probability, events: [n], 0 < p < 1, N >= 1. Sums the pmf over the central range
    holding all but < 1e-12 of the mass.
    """
    from scipy.stats import binom

    p = np.asarray(probability, dtype=float)
    n = np.asarray(events, dtype=float)
    if p.shape != n.shape or p.ndim != 1:
        raise ValueError("probability and events must be matching vectors")
    if not ((p > 0) & (p < 1)).all() or not (n >= 1).all() or not pseudocount > 0:
        raise ValueError("Need 0 < p < 1, N >= 1 and a positive pseudocount")
    mean, variance = np.empty_like(p), np.empty_like(p)
    for i, (pi, ni) in enumerate(zip(p, n, strict=True)):
        low = int(max(0, binom.ppf(1e-13, ni, pi) - 1))
        high = int(min(ni, binom.ppf(1 - 1e-13, ni, pi) + 1))
        m = np.arange(low, high + 1)
        weight = binom.pmf(m, ni, pi)
        weight /= weight.sum()
        log_y = np.log((m + pseudocount) / (ni + 2 * pseudocount))
        mean[i] = weight @ log_y
        variance[i] = weight @ (log_y - mean[i]) ** 2
    return mean, variance


def log_target_offset(
    probability: np.ndarray, events: np.ndarray, pseudocount: float
) -> np.ndarray:
    """E[log((m + a)/(N + 2a))] − log p [n]: pseudocount shift plus Jensen shift."""
    mean, _ = binomial_log_moments(probability, events, pseudocount)
    return mean - np.log(np.asarray(probability, dtype=float))


REFIT_TOLERANCE = 0.01
"""Largest change at convergence: offsets and log-mode noise in log units; identity-mode
noise as the relative change of its standard deviation."""


def resolve_max_refits(max_refits, output_transform, observation_noise="learned"):
    """Largest number of refits: None -> 8 when anything is refitted, else 0.

    Refits correct the log target (log mode) and update binomial point noise
    (observation_noise: binomial); identity mode with learned noise has nothing to refit.
    """
    needed = output_transform == "log" or observation_noise == "binomial"
    if max_refits is None:
        return 8 if needed else 0
    if max_refits < 0:
        raise ValueError("max_refits must be nonnegative")
    if max_refits and not needed:
        raise ValueError("max_refits needs output_transform: log or observation_noise: binomial")
    if observation_noise == "binomial" and not max_refits:
        raise ValueError("observation_noise: binomial needs max_refits > 0")
    return max_refits


@dataclass
class CorrectedFit:
    """Result of :func:`fit_mfgp_corrected`; per-level arrays [n_k] or None."""

    gp: MultiFidelityGP
    offsets: list[np.ndarray | None]
    point_noise: list[np.ndarray | None]
    refits: int
    converged: bool
    max_change: float

    def summary(self) -> dict:
        return dict(
            refits=self.refits,
            converged=self.converged,
            max_change=self.max_change,
            bias_correction=any(v is not None for v in self.offsets),
            point_noise=any(v is not None for v in self.point_noise),
        )


def _check_rate(rate, label):
    invalid = int(((rate <= 0) | (rate >= 1)).sum())
    if invalid:
        raise ValueError(
            f"{label}: {invalid} predicted rates outside (0, 1); no binomial noise or "
            "log-target correction (rates are never clipped)"
        )


def count_noise(
    rate: np.ndarray, events: np.ndarray, *, output_transform: str, pseudocount: float = 0.5
) -> np.ndarray:
    """Binomial count noise [n] of an observed fraction in fitted units, at the rate [n].

    Log mode: Var[log((m + a)/(N + 2a))] (exact); identity: p(1 − p)/N. A rate outside
    (0, 1) raises: it is never clipped.
    """
    rate = np.asarray(rate, dtype=float)
    events = np.asarray(events, dtype=float)
    _check_rate(rate, "count noise")
    if output_transform == "log":
        return binomial_log_moments(rate, events, pseudocount)[1]
    return rate * (1 - rate) / events


def fit_mfgp_corrected(
    X_list: list[np.ndarray],
    Y_list: list[np.ndarray],
    *,
    events: list[np.ndarray | None],
    output_transform: str,
    max_refits: int,
    pseudocount: float = 0.5,
    point_noise: bool = False,
    n_restarts: int = 5,
    **fit_kwargs,
) -> CorrectedFit:
    """MFGP fit with refits that use the rate the previous fit predicts at each point.

    Y_list[k]: [n_k, 1] targets in original units (fit_targets); events[k]: [n_k] event
    counts of an observed-fraction level, None for other levels. The first fit
    (``n_restarts``) has the learned constant noise of each level only. Each refit starts
    from the previous hyperparameters (one optimizer run) and, per observed-fraction level,
    at the rate p the previous fit predicts there:

    * log mode: divides the target by exp(offset), offset = E[log y] − log p
      (log_target_offset), so the targets average to log p (bias correction);
    * ``point_noise``: adds the count noise (count_noise) as a fixed variance per point on
      top of the learned level noise. Identity mode refits only for this.

    Refits stop when nothing changes by more than REFIT_TOLERANCE, or after ``max_refits``.
    A predicted rate outside (0, 1) raises: it is never clipped.
    """
    log = output_transform == "log"
    if len(events) != len(Y_list):
        raise ValueError("events needs one entry (or None) per level")
    if max_refits < 0:
        raise ValueError("max_refits must be nonnegative")
    if point_noise and not max_refits:
        raise ValueError("point_noise needs at least one refit")
    if not log and not point_noise:
        raise ValueError("identity mode refits only for point_noise")
    n = len(Y_list)
    gp = fit_mfgp(
        X_list, Y_list, output_transform=output_transform, n_restarts=n_restarts, **fit_kwargs
    )
    offsets: list[np.ndarray | None] = [None] * n
    noises: list[np.ndarray | None] = [None] * n
    refits, change = 0, float("inf")
    while refits < max_refits and change > REFIT_TOLERANCE:
        targets, change = list(Y_list), 0.0
        for k, counts in enumerate(events):
            if counts is None:
                continue
            counts = np.asarray(counts, dtype=float)
            rate, _ = gp.predict(X_list[k], fidelity=k)
            _check_rate(rate, f"Level {k}")
            if log:
                mean_log, noise = binomial_log_moments(rate, counts, pseudocount)
                offset = mean_log - np.log(rate)
                previous = np.zeros_like(offset) if offsets[k] is None else offsets[k]
                change = max(change, float(np.abs(offset - previous).max()))
                offsets[k] = offset
                targets[k] = Y_list[k] * np.exp(-offset)[:, None]
                if point_noise:
                    previous = np.zeros_like(noise) if noises[k] is None else noises[k]
                    change = max(change, float(np.abs(noise - previous).max()))
            else:
                noise = rate * (1 - rate) / counts
                if noises[k] is None:
                    change = float("inf")
                else:
                    ratio = np.sqrt(noise / noises[k])
                    change = max(change, float(np.abs(ratio - 1).max()))
            if point_noise:
                noises[k] = noise
        gp = fit_mfgp(
            X_list,
            targets,
            output_transform=output_transform,
            n_restarts=1,
            initial_parameters=gp.model.param_array.copy(),
            point_noise=noises if point_noise else None,
            **fit_kwargs,
        )
        refits += 1
    return CorrectedFit(
        gp, offsets, noises, refits, converged=change <= REFIT_TOLERANCE, max_change=change
    )


def fit_mfgp(
    X_list: list[np.ndarray],
    Y_list: list[np.ndarray],
    *,
    kernel: str = "rbf",
    n_restarts: int = 5,
    verbose: bool = False,
    output_transform: str = "identity",
    initial_parameters: np.ndarray | None = None,
    point_noise: list[np.ndarray | None] | None = None,
) -> MultiFidelityGP:
    """Fit a linear multi-fidelity GP with one level per (X, Y) pair, lowest fidelity first.

    X_list[k]: [n_k, D_theta]; Y_list[k]: [n_k, 1]. Log mode requires strictly positive
    targets, already smoothed or floored by the caller. point_noise[k] [n_k]: fixed
    observation variance per point in fitted units (MultiFidelityGP.fit).
    """
    if len(X_list) != len(Y_list) or len(X_list) < 2:
        raise ValueError("Need matching X/Y lists with at least two levels")
    return MultiFidelityGP(
        n_fidelities=len(X_list),
        dim_theta=X_list[0].shape[1],
        kernel=kernel,
        output_transform=output_transform,
    ).fit(
        X_list,
        Y_list,
        n_restarts=n_restarts,
        verbose=verbose,
        initial_parameters=initial_parameters,
        point_noise=point_noise,
    )


def evaluate_mfgp_coverage_from_batch(
    mfgp: MultiFidelityGP,
    holdout_batch: StandardBatch,
    *,
    level: int | None = None,
    n_context: int | None = None,
    seed: int = 12345,
) -> dict[str, np.ndarray | float]:
    """Held-out coverage of one GP level (default: the top) against observed fractions.

    The observed fraction uses the target events of a ``n_context`` split (seed + 1),
    or every event without it. Returns the metrics, θ, y_obs, μ, σ and the fractions
    inside ±1/2/3 σ (Gaussian targets 68.27 / 95.45 / 99.73 %).
    """
    from core.gp_metrics import mfgp_metrics

    _reject_event_only(holdout_batch, "holdout_batch")
    level = mfgp.n_fidelities - 1 if level is None else level
    y_obs = observed_fraction(holdout_batch, n_context=n_context, seed=seed + 1)[0]
    theta = holdout_batch.theta
    mu, var = mfgp.predict(theta, fidelity=level)
    intervals = {k: mfgp.predict_interval(theta, fidelity=level, n_sigma=k) for k in (1, 2, 3)}
    inside = {k: float(((y_obs >= lo) & (y_obs <= hi)).mean()) for k, (lo, hi) in intervals.items()}
    return {
        "metrics": mfgp_metrics(mfgp, theta, y_obs, fidelity=level, seed=seed),
        "theta": theta,
        "y_obs": y_obs,
        "mu": mu,
        "sigma": np.sqrt(var),
        **{f"{k}sigma": inside[k] for k in (1, 2, 3)},
    }


# ---------------------------------------------------------------------------
# Synthetic-data wrappers (known truth from a PseudoDataGenerator).
# ---------------------------------------------------------------------------


def synthetic_levels(
    model: EventSurrogate | ConditionalNeuralProcess | None,
    generator: PseudoDataGenerator,
    levels: list[tuple[str, str]],
    *,
    n_trials: dict[str, int],
    n_events: dict[str, int],
    n_mc_samples: int = 50,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """GP levels from fresh synthetic batches, one batch per fidelity.

    levels: (source, fidelity) pairs, lowest fidelity first, e.g. the paper's recipe
    ``[("model_mean", "lf"), ("model_mean", "hf"), ("observed_fraction", "hf")]``.
    Fidelity k (lf before hf) is drawn with seed ``seed + 100 k`` and split in half
    with seed ``seed + 100 k + 1``, so levels of one fidelity share their targets.
    """
    if generator.mode is InputMode.EVENT_ONLY:
        raise ValueError("MFGP needs θ as input; generator mode=EVENT_ONLY has no θ.")
    fidelities = sorted({fidelity for _, fidelity in levels}, key=["lf", "hf"].index)
    batches = {
        fidelity: generator.generate(
            n_trials=n_trials[fidelity], n_events=n_events[fidelity], seed=seed + 100 * k
        )
        for k, fidelity in enumerate(fidelities)
    }
    split_seed = {fidelity: seed + 100 * k + 1 for k, fidelity in enumerate(fidelities)}
    return levels_from_batches(
        model,
        [
            BatchLevel(source, batches[fidelity], n_events[fidelity] // 2, split_seed[fidelity])
            for source, fidelity in levels
        ],
        n_mc_samples=n_mc_samples,
    )


def evaluate_mfgp_coverage(
    mfgp: MultiFidelityGP,
    generator: PseudoDataGenerator,
    *,
    level: int | None = None,
    n_test_trials: int = 100,
    n_test_events: int = 128,
    seed: int = 12345,
) -> dict[str, np.ndarray | float]:
    """Coverage on ``n_test_trials`` fresh synthetic voxels (half of the events as context)."""
    test = generator.generate(n_trials=n_test_trials, n_events=n_test_events, seed=seed)
    return evaluate_mfgp_coverage_from_batch(
        mfgp, test, level=level, n_context=n_test_events // 2, seed=seed
    )


def _reject_event_only(batch: StandardBatch, label: str) -> None:
    if batch.mode is InputMode.EVENT_ONLY:
        raise ValueError(
            f"{label} must contain θ (mode FULL or DESIGN_ONLY); got mode={batch.mode.value}"
        )
