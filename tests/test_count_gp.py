"""Integer likelihood, probability support, count aggregation and persistence."""

import json

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import binom

pytest.importorskip("GPy")
from core.binomial_gp import BinomialGP, BinomialLogit
from core.count_gp_experiment import run_count_gp
from data.optical_counts import prepare_optical_counts
from schemas.count_gp import CountGPConfig
from tests.test_optical_pipeline import config, fixture_file


def test_binomial_likelihood_matches_counts_and_derivatives():
    likelihood = BinomialLogit()
    f = np.array([-20.0, -6.0, 0.0, 6.0, 20.0])[:, None]
    n = np.array([1500, 5000, 100, 1500, 5000])[:, None]
    m = np.array([0, 13, 50, 1499, 5000])[:, None]
    metadata = {"trials": n}
    np.testing.assert_allclose(
        likelihood.logpdf(f, m, metadata), binom.logpmf(m, n, expit(f)), atol=1e-7
    )
    eps = 1e-5
    numerical = (
        likelihood.logpdf(f + eps, m, metadata) - likelihood.logpdf(f - eps, m, metadata)
    ) / (2 * eps)
    np.testing.assert_allclose(likelihood.dlogpdf_df(f, m, metadata), numerical, atol=1e-6)
    numerical2 = (
        likelihood.dlogpdf_df(f + eps, m, metadata) - likelihood.dlogpdf_df(f - eps, m, metadata)
    ) / (2 * eps)
    np.testing.assert_allclose(likelihood.d2logpdf_df2(f, m, metadata), numerical2, atol=1e-6)
    # Extreme logits and zero/all-hit counts stay finite without label smoothing.
    assert np.isfinite(
        likelihood.logpdf(
            np.array([[-1000.0], [1000.0]]),
            np.array([[0.0], [5000.0]]),
            {"trials": np.array([[1500.0], [5000.0]])},
        )
    ).all()


def test_counts_not_scores_and_roundtrip(tmp_path):
    x = np.linspace(-1, 1, 6)[:, None]
    n = np.array([1500, 5000] * 3)
    m = np.array([0, 2, 3, 8, 1, 0])
    gp = BinomialGP().fit(x, m, n, max_iters=25, n_restarts=1)
    np.testing.assert_array_equal(gp.model.Y[:, 0], m)
    np.testing.assert_array_equal(gp.model.Y_metadata["trials"][:, 0], n)
    mean, var = gp.predict(x)
    assert ((mean > 0) & (mean < 1)).all() and (var >= 0).all()
    samples = gp.predict_observations(x, n, n_draws=2048)
    assert ((samples >= 0) & (samples <= 1)).all()
    np.testing.assert_allclose(samples * n, np.round(samples * n), atol=1e-10)
    restored = BinomialGP.load(gp.save(tmp_path / "model.pkl"))
    np.testing.assert_array_equal(mean, restored.predict(x)[0])
    np.testing.assert_array_equal(samples, restored.predict_observations(x, n, n_draws=2048))
    with pytest.raises(ValueError, match="integers"):
        BinomialGP().fit(x, m + 0.1, n)
    with pytest.raises(ValueError, match="exceed"):
        BinomialGP().fit(x, n + 1, n)


def test_more_trials_constrain_same_rate_more():
    x = np.array([[0.0], [1.0], [2.0]])
    low = BinomialGP().fit(x, [1, 1, 1], [100, 100, 100], max_iters=5, n_restarts=1)
    high = BinomialGP().fit(x, [100, 100, 100], [10000] * 3, max_iters=5, n_restarts=1)
    # Isolate the likelihood's N effect from hyperparameter optimization.
    high.model[:] = low.model.param_array.copy()
    assert np.all(high.predict_latent(x)[1] < low.predict_latent(x)[1])


def test_optical_counts_end_to_end(tmp_path):
    c = config(tmp_path)
    for i in range(4):
        fixture_file(
            c.source.directory,
            "lf",
            i,
            center=(float(i), 0.0, 0.0),
            n=8,
            hit_ids=() if i == 0 else (3, 3, 1),
            detectors=() if i == 0 else (10, 11, 10),
        )
    for i in range(5):
        fixture_file(c.source.directory, "hf", i, center=(float(i), 1.0, 0.0), n=12)
    c.split.lf_train_only = True
    c.split.hf_train_count = 2
    prepared = prepare_optical_counts(c)
    data, rows, _ = prepared
    assert sum(r["hits"] == 0 for r in rows) == 1
    assert {r["trials"] for r in rows} == {8, 12}
    assert all(r["hits"] in (0, 2) for r in rows)  # Multiple hits count one event.
    cfg = CountGPConfig(
        data=c,
        output_directory=tmp_path / "run",
        backend={"max_iters": 25, "n_restarts": 1},
        prediction_draws=2048,
    )
    out = run_count_gp(cfg, prepared)
    gp = BinomialGP.load(out / "model.pkl")
    np.testing.assert_allclose(gp.offset, data["train"]["theta"].mean(0))
    assert (out / "hf_coverage.png").exists()
    rows = json.loads((out / "metrics.json").read_text())
    assert len(rows) == 1
    assert rows[0]["mean_bias"] == pytest.approx(
        rows[0]["mean_predicted"] - rows[0]["mean_observed"]
    )
    assert rows[0]["crps"] >= 0
    assert rows[0]["predictive_nll"] >= 0
    assert "mean_width" in rows[0]["interval_metrics"]["2"]
    with np.load(out / "hf_validation.npz") as predictions:
        assert (predictions["trials"] == 12).all()
        assert (predictions["lower_3"] >= 0).all()
    # A source change is automatically reflected without an event/context split.
    fixture_file(c.source.directory, "lf", 10, center=(10.0, 0.0, 0.0), n=15)
    updated = prepare_optical_counts(c)[0]
    assert 15 in updated["train"]["trials"]


def test_prediction_covariance_with_large_nearly_constant_kernel():
    """Regression: inverse-based prediction lost PSD on the all-budget data."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, (100, 3))
    n = np.full(len(x), 1500)
    m = rng.binomial(n, expit(-6 + x[:, 2]))
    gp = BinomialGP().fit(x, m, n, max_iters=5, n_restarts=1)
    gp.model[:] = [9992.0, 5735.0, 6811.0, 7.48, 4105.0]
    query = rng.uniform(-1, 1, (30, 3))
    mean, covariance = gp.predict_latent(query, full_cov=True)
    _, diagonal = gp.predict_latent(query)
    np.testing.assert_allclose(diagonal, np.diag(covariance), atol=1e-9)
    assert np.linalg.eigvalsh(covariance).min() > -1e-9
    assert np.isfinite(mean).all()
    samples = gp.predict_observations(query, 5000, n_draws=2048)
    assert np.isfinite(samples).all()
