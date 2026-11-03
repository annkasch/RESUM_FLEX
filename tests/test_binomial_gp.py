"""Binomial count GP: integer likelihood, persistence, the runner and its config."""

import json

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import binom

pytest.importorskip("GPy")
from core.binomial_gp import BinomialGP, BinomialLogit  # noqa: E402
from data.pseudo_generator import for_scenario  # noqa: E402
from schemas.surrogates import BinomialGPRunConfig  # noqa: E402


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
    extreme = likelihood.logpdf(
        np.array([[-1000.0], [1000.0]]),
        np.array([[0.0], [5000.0]]),
        {"trials": np.array([[1500.0], [5000.0]])},
    )
    assert np.isfinite(extreme).all()  # zero / all-hit counts without label smoothing


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
    np.testing.assert_allclose(samples * n, np.round(samples * n), atol=1e-10)
    restored = BinomialGP.load(gp.save(tmp_path / "model.pkl"))
    np.testing.assert_array_equal(mean, restored.predict(x)[0])


def write_batches(root):
    source = for_scenario("S1", seed=2)
    for split, fidelity, n_events, seed in [
        ("train", "lf", 30, 1),
        ("train", "hf", 80, 2),
        ("validation", "hf", 80, 3),
    ]:
        batch = source.generate(n_trials=6, n_events=n_events, seed=seed)
        folder = root / "batches" / split
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{fidelity}.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )
    return root


def test_runner_pools_fidelities_and_scores_evaluation(tmp_path):
    from core.surrogates.binomial_stage import run_binomial_config

    root = write_batches(tmp_path / "data")
    config = BinomialGPRunConfig(
        observations=[
            {"partition": "train", "fidelity": "lf", "max_voxels": 4},
            {"partition": "train", "fidelity": "hf"},
        ],
        evaluation=[{"partition": "validation", "fidelity": "hf"}],
        n_restarts=1,
        max_iters=30,
        prediction_draws=2048,
        data_directory=root,
        output_directory=tmp_path / "gp",
    )
    out = run_binomial_config(config)
    sets = json.loads((out / "observations.json").read_text())
    assert [(s["partition"], s["fidelity"], s["voxels"]) for s in sets] == [
        ("train", "lf", 4),
        ("train", "hf", 6),
    ]
    with np.load(out / "training_arrays.npz") as arrays:
        assert len(arrays["hits"]) == 10 and set(arrays["events"]) == {30, 80}
    row = json.loads((out / "metrics.json").read_text())[0]
    assert row["voxels"] == 6 and np.isfinite(row["predictive_nll"])
    assert set(row["coverage"]) == {"1", "2", "3"}
    with np.load(out / "evaluation_0.npz") as arrays:
        assert (arrays["lower_1"] <= arrays["upper_1"]).all()
    for name in ("evaluation_0_coverage.png", "evaluation_0_zscore.png", "model.pkl"):
        assert (out / name).exists()
    with pytest.raises(FileExistsError):
        run_binomial_config(config)


def test_binomial_config_blinds_test_partitions():
    base = dict(
        observations=[{"partition": "train", "fidelity": "hf"}],
        data_directory="d",
        output_directory="o",
    )
    with pytest.raises(ValueError, match="blinded"):
        BinomialGPRunConfig(**base, evaluation=[{"partition": "gp_test"}])
    with pytest.raises(ValueError, match="can't be a training set"):
        BinomialGPRunConfig(**{**base, "observations": [{"partition": "test", "fidelity": "hf"}]})
    assert BinomialGPRunConfig(**base, evaluation=[{"partition": "gp_test"}], unblind_test=True)


def test_coverage_ylim_config():
    base = dict(
        observations=[{"partition": "train", "fidelity": "hf"}],
        data_directory="d",
        output_directory="o",
    )
    assert BinomialGPRunConfig(**base, coverage_ylim=(0, 0.0065)).coverage_ylim == (0, 0.0065)
    assert BinomialGPRunConfig(**base).coverage_ylim is None
    with pytest.raises(ValueError, match="low < high"):
        BinomialGPRunConfig(**base, coverage_ylim=(0.01, 0.0))
