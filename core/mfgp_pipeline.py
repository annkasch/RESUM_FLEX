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
            y = model_mean(model, level.batch, n_context=level.n_context, seed=level.seed,
                           n_mc_samples=n_mc_samples)
        else:
            y = observed_fraction(level.batch, n_context=level.n_context, seed=level.seed)[0]
        X_list.append(level.batch.theta)
        Y_list.append(np.asarray(y, dtype=float).reshape(-1, 1))
    return X_list, Y_list


def fit_mfgp(
    X_list: list[np.ndarray],
    Y_list: list[np.ndarray],
    *,
    kernel: str = "rbf",
    n_restarts: int = 5,
    verbose: bool = False,
    output_transform: str = "identity",
) -> MultiFidelityGP:
    """Fit a linear multi-fidelity GP with one level per (X, Y) pair, lowest fidelity first.

    X_list[k]: [n_k, D_theta]; Y_list[k]: [n_k, 1]. Log mode requires strictly positive
    targets, already smoothed or floored by the caller.
    """
    if len(X_list) != len(Y_list) or len(X_list) < 2:
        raise ValueError("Need matching X/Y lists with at least two levels")
    return MultiFidelityGP(
        n_fidelities=len(X_list),
        dim_theta=X_list[0].shape[1],
        kernel=kernel,
        output_transform=output_transform,
    ).fit(X_list, Y_list, n_restarts=n_restarts, verbose=verbose)


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
