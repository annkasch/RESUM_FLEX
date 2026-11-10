"""Randomized quantile residuals of hit counts per design point (Dunn & Smyth 1996).

For a count m with predictive CDF F, u ~ Uniform(F(m − 1), F(m)) and z = Φ⁻¹(u). Under a
correct predictive z is exactly N(0, 1) at any expected count, unlike the Pearson residual
(m − E m) / sd, which is discrete and skewed when the expected hits per point are ~1.

Every predictive enters through four logs per point [B]: log F(m − 1), log F(m),
log S(m − 1), log S(m) with S = 1 − F. The draw is made in whichever tail is smaller, so
z stays accurate far into both tails; a count with zero predicted mass gives z = ±inf.

Predictives:
  * poisson_binomial_log_tails: hits of independent events with probabilities p (event
    models), exact up to EXACT_MAX_HITS, normal with continuity correction above.
  * binomial_mixture_log_tails: Binomial(N, p) mixed over draws of p (binomial GP).
  * discretized_gaussian_log_tails: a Gaussian on a monotone transform of m (MFGP),
    discretized at half-integer counts.
"""

import numpy as np
from scipy.special import logsumexp, ndtri, ndtri_exp
from scipy.stats import binom, norm

# Exact Poisson-binomial recursion for counts up to this; larger counts use the normal
# approximation (skewness ≲ 1/√200, well below the randomization noise of one point).
EXACT_MAX_HITS = 200

_LOG_HALF = np.log(0.5)


def _log_mix(log_low, log_high, v):
    """log(low + v (high − low)) from log low ≤ log high; v in [0, 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.exp(log_low - log_high)  # [B] in [0, 1]; 0 when low = 0
        ratio = np.where(np.isneginf(log_high), 0.0, ratio)
        return log_high + np.log(v + (1 - v) * ratio)


def quantile_residuals(log_cdf_below, log_cdf, log_sf_below, log_sf, rng):
    """z [B] = Φ⁻¹(u), u ~ U(F(m − 1), F(m)); arguments are the four log tails [B].

    log_cdf_below = log F(m − 1) (−inf for m = 0), log_cdf = log F(m), log_sf_below =
    log S(m − 1), log_sf = log S(m) (−inf for m = N). u is drawn in the lower tail when
    F(m) ≤ 1/2, in the upper tail when S(m − 1) ≤ 1/2, else directly.
    """
    arrays = [np.asarray(a, dtype=float) for a in (log_cdf_below, log_cdf, log_sf_below, log_sf)]
    a, b, sa, sb = arrays
    if len({x.shape for x in arrays}) != 1 or a.ndim != 1:
        raise ValueError("The four log tails must be vectors of equal length")
    if np.isnan(arrays).any() or (np.asarray(arrays) > 1e-12).any():
        raise ValueError("Log tail probabilities must be ≤ 0 and not NaN")
    v = rng.random(len(a))  # [B]
    z = np.empty(len(a))
    lower = b <= _LOG_HALF
    upper = ~lower & (sa <= _LOG_HALF)
    middle = ~lower & ~upper
    with np.errstate(divide="ignore"):
        z[lower] = ndtri_exp(_log_mix(a[lower], b[lower], v[lower]))
        # 1 − u = S(m) + (1 − v)(S(m − 1) − S(m)), so the same mix from the upper tail.
        z[upper] = -ndtri_exp(_log_mix(sb[upper], sa[upper], 1 - v[upper]))
        u = np.exp(a[middle]) + v[middle] * (np.exp(b[middle]) - np.exp(a[middle]))
        z[middle] = ndtri(u)
    return z


def poisson_binomial_log_tails(hits, probabilities, mask):
    """Four log tails [B] of hits [B] under independent Bernoulli(p) events.

    probabilities, mask: [B, N]; padded events (mask False) are ignored. Exact for
    hits ≤ EXACT_MAX_HITS: a recursion over events that keeps the mass of 0..m hits and of
    "more than m" separately, rescaled at every event so nothing underflows. Larger counts:
    normal approximation with continuity correction.
    """
    hits = np.asarray(hits)
    p = np.where(mask, probabilities, 0.0)  # [B, N]; p = 0 leaves the counts unchanged
    if hits.shape != (len(p),) or (hits < 0).any() or (hits > mask.sum(axis=1)).any():
        raise ValueError("hits must be [B] counts between 0 and the valid events per point")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Event probabilities must lie in [0, 1]")
    out = np.empty((4, len(hits)))
    exact = hits <= EXACT_MAX_HITS
    # Points with similar counts share one vectorized recursion of width 2^k + 2.
    width = np.where(exact, 2 ** np.ceil(np.log2(hits + 1)).astype(int), 0)
    for w in np.unique(width[exact]):
        index = np.flatnonzero(width == w)
        out[:, index] = _exact_tails(hits[index], p[index], int(w))
    if (~exact).any():
        mean = p[~exact].sum(axis=1)
        sd = np.sqrt((p[~exact] * (1 - p[~exact])).sum(axis=1))
        m = hits[~exact]
        high, low = (m + 0.5 - mean) / sd, (m - 0.5 - mean) / sd
        out[:, ~exact] = norm.logcdf(low), norm.logcdf(high), norm.logsf(low), norm.logsf(high)
    return tuple(out)


def _exact_tails(hits, p, width):
    """Log tails [4, b] for hits [b] ≤ width − 1; p [b, N]."""
    b = len(hits)
    # states 0..width-1 hold exact counts, state `width` holds "more than width − 1".
    mass = np.zeros((b, width + 1))
    mass[:, 0] = 1.0
    log_scale = np.zeros(b)
    for column in p.T:  # [b]
        q = column[:, None]
        shifted = np.zeros_like(mass)
        shifted[:, 1:] = mass[:, :-1]
        shifted[:, -1] += mass[:, -1]  # a hit keeps the overflow state overflowed
        mass = mass * (1 - q) + shifted * q
        top = mass.max(axis=1)  # [b] > 0: the mass sums to one before rescaling
        mass /= top[:, None]
        log_scale += np.log(top)
    rows, previous = np.arange(b), np.maximum(hits - 1, 0)
    cumulative = np.cumsum(mass[:, :-1], axis=1)  # F(k) [b, width], k = 0..width-1
    # S(k) = mass of all states above k, overflow included: no 1 − F cancellation.
    above = mass[:, ::-1].cumsum(axis=1)[:, ::-1]  # [b, width + 1], states ≥ k
    tail = above[:, 1:]  # S(k) [b, width]
    below = np.where(hits > 0, cumulative[rows, previous], 0.0)
    sf_below = np.where(hits > 0, tail[rows, previous], above[:, 0])
    with np.errstate(divide="ignore"):
        return np.stack(
            [
                np.log(below) + log_scale,
                np.log(cumulative[rows, hits]) + log_scale,
                np.log(sf_below) + log_scale,
                np.log(tail[rows, hits]) + log_scale,
            ]
        )


def binomial_mixture_log_tails(hits, events, p_draws):
    """Four log tails [B] of hits [B] under Binomial(events, p) mixed over p_draws [D, B]."""
    hits, events = np.asarray(hits), np.asarray(events)
    p = np.asarray(p_draws, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(hits) or hits.shape != events.shape:
        raise ValueError("p_draws must be [draws, B] for hits and events [B]")
    log_d = np.log(len(p))

    def mix(values):
        return logsumexp(values, axis=0) - log_d

    with np.errstate(divide="ignore"):
        return (
            mix(binom.logcdf(hits - 1, events, p)),
            mix(binom.logcdf(hits, events, p)),
            mix(binom.logsf(hits - 1, events, p)),
            mix(binom.logsf(hits, events, p)),
        )


def discretized_gaussian_log_tails(hits, events, mean, sigma, transform):
    """Four log tails [B] of hits [B] when transform(count, events) ~ N(mean, sigma²).

    transform: monotone increasing in the count, applied at the half-integer edges
    m ± 1/2; all Gaussian mass below the edge of 0 goes to 0 hits, above that of N to N.
    """
    hits, events = np.asarray(hits, dtype=float), np.asarray(events, dtype=float)
    mean, sigma = np.asarray(mean, dtype=float), np.asarray(sigma, dtype=float)
    if not (sigma > 0).all():
        raise ValueError("The Gaussian predictive needs sigma > 0")
    low = (transform(np.maximum(hits - 0.5, 0.0), events) - mean) / sigma
    high = (transform(np.minimum(hits + 0.5, events), events) - mean) / sigma
    first, last = hits == 0, hits == events
    return (
        np.where(first, -np.inf, norm.logcdf(low)),
        np.where(last, 0.0, norm.logcdf(high)),
        np.where(first, 0.0, norm.logsf(low)),
        np.where(last, -np.inf, norm.logsf(high)),
    )
