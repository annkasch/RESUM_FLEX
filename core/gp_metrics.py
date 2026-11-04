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


# Central predictive intervals keyed by Gaussian-equivalent sigma level.
SIGMA_LEVELS = {1: erf(1 / sqrt(2)), 2: erf(2 / sqrt(2)), 3: erf(3 / sqrt(2))}


def count_predictive_scores(hits, trials, probability_draws, *, seed=0):
    """Score a count prediction given draws of each voxel's hit probability.

    hits, trials: [B] observed counts; probability_draws: [S, B] equal-weight draws of p.
    The observation predictive is k ~ Binomial(N, p) mixed over the draws: its fractions
    give intervals, coverage and CRPS (gp_metrics), and the count NLL is the binomial mass
    mixed over the same draws. Returns (scores, arrays) like the binomial GP evaluation.
    """
    from scipy.special import logsumexp
    from scipy.stats import binom

    k, n = np.asarray(hits), np.asarray(trials)
    p = np.asarray(probability_draws, float)
    if k.ndim != 1 or n.shape != k.shape or p.ndim != 2 or p.shape[1] != len(k):
        raise ValueError("Expected hits, trials [B] and probability_draws [S, B]")
    if np.any(n < 1) or np.any(k < 0) or np.any(k > n):
        raise ValueError("Counts need 0 <= hits <= trials and trials >= 1")
    if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("Probability draws must lie strictly inside (0, 1)")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    observed = k / n
    samples = rng.binomial(n, p) / n  # [S, B] predicted fractions
    intervals = {
        level: tuple(np.quantile(samples, [(1 - q) / 2, (1 + q) / 2], axis=0))
        for level, q in SIGMA_LEVELS.items()
    }
    mean = samples.mean(axis=0)
    scores = gp_metrics(
        observed, mean, intervals=intervals, samples=samples, median=np.median(samples, axis=0)
    )
    log_mass = logsumexp(binom.logpmf(k, n, p), axis=0) - np.log(len(p))
    scores.update(
        predictive_nll=float(-log_mass.mean()),
        predictive_distribution="binomial counts mixed over draws of p",
        nll_measure="count probability mass; not comparable to a density NLL",
        coverage={
            str(level): float(((observed >= lo) & (observed <= hi)).mean())
            for level, (lo, hi) in intervals.items()
        },
        draws=len(p),
    )
    arrays = dict(
        observed=observed,
        hits=k,
        target_events=n,
        mean=mean,
        sigma=samples.std(axis=0),
        probability_mean=p.mean(axis=0),
        probability_sd=p.std(axis=0),
        **{
            f"{bound}_{level}": intervals[level][i]
            for level in SIGMA_LEVELS
            for i, bound in enumerate(("lower", "upper"))
        },
    )
    return scores, arrays
