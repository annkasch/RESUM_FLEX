"""Shared held-out GP metrics in original response units, equally weighted by voxel."""

from math import erf, sqrt

import numpy as np
from scipy.stats import lognorm, norm, rankdata


def gp_metrics(observed, predicted, *, intervals=None, samples=None, median=None):
    """Evaluate supplied predictions; never infer a likelihood from error bars.

    Intervals are equal-tail bands keyed by Gaussian-equivalent sigma level.
    Samples have shape (draws, voxels). Undefined correlations are JSON null.
    Calibration fits observed = intercept + slope * predicted (descriptive OLS).
    """
    y, p = np.asarray(observed, float), np.asarray(predicted, float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Observed and predicted must be matching nonempty vectors")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite observations and predictions")
    residual = p - y

    def correlation(a, b):
        if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    slope = (
        None
        if np.ptp(p) == 0
        else float(np.dot(p - p.mean(), y - y.mean()) / np.sum((p - p.mean()) ** 2))
    )
    result = dict(
        voxels=len(y),
        weighting="equal per validation voxel",
        response_units="original response",
        mean_observed=float(y.mean()),
        mean_predicted=float(p.mean()),
        mean_bias=float(residual.mean()),
        mae=float(np.abs(residual).mean()),
        rmse=float(np.sqrt(np.mean(residual**2))),
        pearson_r=correlation(p, y),
        spearman_r=correlation(rankdata(p), rankdata(y)),
        calibration_slope=slope,
        calibration_intercept=None if slope is None else float(y.mean() - slope * p.mean()),
        calibration_note=(
            "Descriptive OLS: observed = intercept + slope * predicted; noisy observations"
        ),
    )
    if samples is not None:
        samples = np.asarray(samples, float)
        if (
            samples.ndim != 2
            or samples.shape[1] != len(y)
            or not len(samples)
            or not np.isfinite(samples).all()
        ):
            raise ValueError("Predictive samples must be finite with shape (draws, voxels)")
        ordered = np.sort(samples, axis=0)
        n = len(ordered)
        # Empirical CRPS = E|X-y| - E|X-X'|/2, without a quadratic pair matrix.
        correction = ((2 * np.arange(1, n + 1) - n - 1)[:, None] * ordered).sum(0) / n**2
        result["crps"] = float((np.abs(samples - y).mean(0) - correction).mean())
        if median is None:
            median = np.median(samples, axis=0)
    details, weighted_scores = {}, []
    for k, bounds in (intervals or {}).items():
        lo, hi = map(lambda a: np.asarray(a, float), bounds)
        if (
            lo.shape != y.shape
            or hi.shape != y.shape
            or not np.isfinite([lo, hi]).all()
            or np.any(lo > hi)
        ):
            raise ValueError("Invalid interval bounds")
        mass = erf(float(k) / sqrt(2))
        alpha = 1 - mass
        width = hi - lo
        score = width + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
        inside = int(((y >= lo) & (y <= hi)).sum())
        details[str(k)] = dict(
            nominal_probability=mass,
            inside=inside,
            total=len(y),
            coverage=inside / len(y),
            mean_width=float(width.mean()),
            interval_score=float(score.mean()),
        )
        weighted_scores.append(alpha / 2 * score)
    result["interval_metrics"] = details
    if weighted_scores and median is not None:
        median = np.asarray(median, float)
        if median.shape != y.shape or not np.isfinite(median).all():
            raise ValueError("Predictive median must match observations")
        wis = (0.5 * np.abs(y - median) + np.sum(weighted_scores, axis=0)) / (
            len(weighted_scores) + 0.5
        )
        result["weighted_interval_score"] = float(wis.mean())
    return result


def mfgp_metrics(gp, theta, observed, *, fidelity=None, seed=42, n_draws=4096):
    """Score the actual Gaussian/lognormal observation distribution of an MFGP."""
    mean, _ = gp.predict(theta, fidelity=fidelity)
    mu, variance = gp.predict_transformed(theta, fidelity=fidelity)
    sigma = np.sqrt(variance)
    intervals = {k: gp.predict_interval(theta, fidelity=fidelity, n_sigma=k) for k in (1, 2, 3)}
    draws = np.random.default_rng(seed).normal(mu, sigma, size=(n_draws, len(mu)))
    log_space = getattr(gp, "output_transform", "identity") == "log"
    samples = np.exp(draws) if log_space else draws
    median = np.exp(mu) if log_space else mu
    result = gp_metrics(observed, mean, intervals=intervals, samples=samples, median=median)
    logpdf = (
        lognorm.logpdf(observed, s=sigma, scale=np.exp(mu))
        if log_space
        else norm.logpdf(observed, loc=mu, scale=sigma)
    )
    result["predictive_nll"] = float(-logpdf.mean()) if np.isfinite(logpdf).all() else None
    result["predictive_nll_status"] = (
        "finite"
        if np.isfinite(logpdf).all()
        else "nonfinite: observation outside support or degenerate variance"
    )
    result["predictive_distribution"] = (
        "lognormal response density" if log_space else "Gaussian response density"
    )
    result["nll_measure"] = (
        "density in original response units; not comparable to count probability mass"
    )
    result["crps_method"] = f"empirical marginal predictive draws; n={n_draws}, seed={seed}"
    return result
