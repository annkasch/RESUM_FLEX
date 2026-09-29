"""Fixed-episode CNP diagnostics for observed data without analytical truth."""

import numpy as np
import torch

from core.surrogate_cnp import split_context_target
from core.training import cnp_trial_predictive


def coverage_counts(observed, predicted, sigma):
    """Count distinct held-out trials, not Monte Carlo draws or repeated episodes."""
    observed, predicted, sigma = [np.asarray(a, dtype=float) for a in (observed, predicted, sigma)]
    if (
        observed.ndim != 1
        or not len(observed)
        or not (observed.shape == predicted.shape == sigma.shape)
    ):
        raise ValueError("Expected matching nonempty 1-D arrays")
    if not all(np.isfinite(a).all() for a in (observed, predicted, sigma)) or np.any(sigma < 0):
        raise ValueError("Coverage inputs must be finite with nonnegative sigma")
    error = np.abs(observed - predicted)
    return [
        {
            "sigma": k,
            "within": int((error <= k * sigma).sum()),
            "total": len(observed),
            "fraction": float((error <= k * sigma).mean()),
        }
        for k in (1, 2, 3)
    ]


def evaluate_cnp_validation(model, batch, *, n_context=64, n_mc_samples=200, seed=12345):
    """Evaluate all target events in one fixed context/target partition per voxel.

    Uses RESUM_FLEX's existing approximate trial uncertainty, including sampling
    noise. Coverage is against observed target m/N, not unknown latent truth.
    """
    if not 1 <= n_context < batch.n_events:
        raise ValueError("n_context must leave at least one target event")
    if n_mc_samples < 2:
        raise ValueError("n_mc_samples must be at least two")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("This fixed-data validation helper currently supports CPU models")
    ctx, tgt = split_context_target(batch, n_context=n_context, seed=seed)
    previous_mode = model.training
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            pred = cnp_trial_predictive(
                model, ctx, tgt, n_mc_samples=n_mc_samples, include_aleatoric=True
            )
            with torch.no_grad():
                probabilities = model.predict_beta(ctx, tgt).cpu().numpy().astype(float)
    finally:
        model.train(previous_mode)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite event predictions")
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    y = tgt.labels
    observed = y.mean(1)
    coverage = coverage_counts(observed, pred["y_cnp"], pred["sigma_total"])
    residual = observed - pred["y_cnp"]
    pulls = np.divide(
        residual,
        pred["sigma_total"],
        out=np.full_like(residual, np.nan, dtype=float),
        where=pred["sigma_total"] > 0,
    )
    summary = {
        "voxels": batch.batch_size,
        "context_events_per_voxel": n_context,
        "target_events_per_voxel": tgt.n_events,
        "target_positive_events": int(y.sum()),
        "bernoulli_log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
        "brier_score": float(np.mean((p - y) ** 2)),
        "voxel_rate_mae": float(np.mean(np.abs(residual))),
        "voxel_rate_rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_signed_residual": float(residual.mean()),
        "mean_sigma_total": float(pred["sigma_total"].mean()),
        "zero_sigma_voxels": int((pred["sigma_total"] == 0).sum()),
        "seed": seed,
        "n_mc_samples": n_mc_samples,
        "coverage": coverage,
    }
    return {
        "summary": summary,
        "observed": observed,
        "predicted": pred["y_cnp"],
        "sigma_total": pred["sigma_total"],
        "sigma_epistemic": pred["sigma_epistemic"],
        "sigma_aleatoric": pred["sigma_aleatoric"],
        "pull": pulls,
        "target_positive_events": y.sum(1),
    }
