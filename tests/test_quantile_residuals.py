"""Randomized quantile residuals: exact tails, N(0, 1) at low counts, extreme tails."""

import numpy as np
import pytest
from scipy.stats import binom, kstest, norm

from core.quantile_residuals import (
    EXACT_MAX_HITS,
    binomial_mixture_log_tails,
    discretized_gaussian_log_tails,
    poisson_binomial_log_tails,
    quantile_residuals,
)


def _binomial_tails(m, n, p):
    return (
        binom.logcdf(m - 1, n, p),
        binom.logcdf(m, n, p),
        binom.logsf(m - 1, n, p),
        binom.logsf(m, n, p),
    )


def test_poisson_binomial_matches_binomial_and_ignores_padding():
    rng = np.random.default_rng(0)
    m = np.concatenate([rng.integers(0, 15, 30), [0, 300]])  # 300: normal approximation
    p = np.full((32, 400), 0.02)
    mask = np.ones_like(p, dtype=bool)
    mask[:, 300:] = False  # 300 valid events per point
    p[~mask] = 0.7  # padded probabilities must not count
    tails = poisson_binomial_log_tails(m, p, mask)
    exact = m <= EXACT_MAX_HITS
    for got, want in zip(tails, _binomial_tails(m, 300, 0.02), strict=True):
        np.testing.assert_allclose(np.exp(got[exact]), np.exp(want[exact]), atol=1e-12)
    assert np.isneginf(tails[0][m == 0]).all()


def test_poisson_binomial_far_tails_without_underflow():
    n, p = 20000, 1e-3  # 20 expected hits
    tails = poisson_binomial_log_tails(
        np.array([0, 150]), np.full((2, n), p), np.ones((2, n), dtype=bool)
    )
    assert tails[1][0] == pytest.approx(binom.logcdf(0, n, p), rel=1e-9)
    assert tails[3][1] == pytest.approx(binom.logsf(150, n, p), rel=1e-9)  # ~ e^-178
    z = quantile_residuals(*tails, np.random.default_rng(1))
    assert np.isfinite(z).all() and z[0] < -4 and z[1] > 18  # Φ⁻¹(1 − e^-178) ≈ 18.7


@pytest.mark.parametrize("scale", [0.003, 0.05])  # ~0.3 and ~5 expected hits per point
def test_quantile_residuals_are_standard_normal_where_pearson_is_not(scale):
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 2 * scale, (3000, 100))
    counts = rng.integers(40, 101, 3000)
    mask = np.arange(100)[None] < counts[:, None]
    hits = ((rng.random(p.shape) < p) & mask).sum(axis=1)
    z = quantile_residuals(*poisson_binomial_log_tails(hits, p, mask), rng)
    assert kstest(z, "norm").pvalue > 1e-3
    q = np.where(mask, p, 0)
    pearson = (hits - q.sum(axis=1)) / np.sqrt((q * (1 - q)).sum(axis=1))
    assert kstest(pearson, "norm").pvalue < 1e-3


def test_wrong_rates_are_detected():
    rng = np.random.default_rng(3)
    p = np.full((2000, 200), 0.01)
    hits = (rng.random(p.shape) < 2 * p).sum(axis=1)  # truth twice the prediction
    z = quantile_residuals(*poisson_binomial_log_tails(hits, p, np.ones_like(p, bool)), rng)
    assert z.mean() > 0.5


def test_binomial_mixture_calibrated_and_matches_single_draw():
    rng = np.random.default_rng(4)
    n = np.full(2000, 150)
    draws = rng.beta(1.5, 300, (500, 2000))  # [D, B]
    hits = rng.binomial(n, draws[rng.integers(0, 500, 2000), np.arange(2000)])
    z = quantile_residuals(*binomial_mixture_log_tails(hits, n, draws), rng)
    assert kstest(z, "norm").pvalue > 1e-3
    single = binomial_mixture_log_tails(hits[:5], n[:5], draws[:1, :5])
    for got, want in zip(single, _binomial_tails(hits[:5], 150, draws[0, :5]), strict=True):
        np.testing.assert_allclose(got, want)


def test_discretized_gaussian_puts_outer_mass_on_the_end_counts():
    n = np.array([10.0, 10.0])
    hits = np.array([0.0, 10.0])
    tails = discretized_gaussian_log_tails(
        hits, n, np.array([0.0, 1.0]), np.array([0.1, 0.1]), lambda count, events: count / events
    )
    np.testing.assert_allclose(np.exp(tails[1][0]), norm.cdf(0.05 / 0.1))  # F(0) incl. < 0
    np.testing.assert_allclose(np.exp(tails[2][1]), norm.sf(-0.05 / 0.1))  # S(9) incl. > 1
    assert np.isneginf(tails[0][0]) and np.isneginf(tails[3][1])


def test_quantile_residuals_seeded_and_validated():
    tails = _binomial_tails(np.array([0, 3, 9]), 20, 0.2)
    first = quantile_residuals(*tails, np.random.default_rng(5))
    np.testing.assert_array_equal(first, quantile_residuals(*tails, np.random.default_rng(5)))
    with pytest.raises(ValueError, match="≤ 0"):
        quantile_residuals(
            np.zeros(1), np.ones(1), np.zeros(1), np.zeros(1), np.random.default_rng()
        )
    with pytest.raises(ValueError, match="hits"):
        poisson_binomial_log_tails(np.array([5]), np.full((1, 3), 0.5), np.ones((1, 3), bool))
