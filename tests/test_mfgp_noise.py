"""Fixed per-voxel (binomial) observation noise in the multi-fidelity GP."""

import json

import numpy as np
import pytest

pytest.importorskip("GPy")

from core.surrogate_mfgp import MultiFidelityGP, binomial_noise, load_mfgp, save_mfgp  # noqa: E402
from schemas.surrogates import MFGPRunConfig  # noqa: E402


@pytest.mark.parametrize("transform", ["identity", "log"])
def test_binomial_noise_matches_monte_carlo(transform):
    rng = np.random.default_rng(0)
    p, n, a = np.array([0.02, 0.1, 0.4]), np.array([1500, 5000, 1500]), 0.5
    smoothed = (rng.binomial(n, p, size=(200_000, 3)) + a) / (n + 2 * a)
    values = np.log(smoothed) if transform == "log" else smoothed
    expected = binomial_noise(p, n, output_transform=transform, pseudocount=a)
    # Log: first-order delta method, a few % low at ~30 expected hits.
    np.testing.assert_allclose(
        values.var(axis=0), expected, rtol=0.1 if transform == "log" else 0.02
    )
    # Dim voxels are noisier in log space; no hits still gives a finite variance.
    assert np.isfinite(
        binomial_noise(
            np.array([a / 1501]), np.array([1500]), output_transform="log", pseudocount=a
        )
    ).all()
    with pytest.raises(ValueError, match="probabilities"):
        binomial_noise(np.array([1.2]), np.array([10]), output_transform=transform, pseudocount=a)


def two_levels(seed=0):
    rng = np.random.default_rng(seed)
    x_lf = np.linspace(0, 1, 25)[:, None]
    x_hf = np.linspace(0, 1, 9)[:, None]
    truth = lambda x: np.sin(6 * x)  # noqa: E731
    return [x_lf, x_hf], [
        truth(x_lf) + 0.05 * rng.normal(size=x_lf.shape),
        1.5 * truth(x_hf) + 0.01 * rng.normal(size=x_hf.shape),
    ]


def test_fixed_noise_enters_the_likelihood(tmp_path):
    from scipy.stats import multivariate_normal

    X, Y = two_levels()
    noise = np.linspace(0.01, 0.5, len(X[1]))
    np.random.seed(0)
    gp = MultiFidelityGP(2, 1, kernel="rbf").fit(X, Y, n_restarts=1, noise_list=[None, noise])
    model = gp.model
    learned = [lik.variance[0] for lik in model.likelihood.likelihoods_list]
    total = np.r_[np.full(len(X[0]), learned[0]), learned[1] + noise]
    np.testing.assert_allclose(model.likelihood.gaussian_variance(model.Y_metadata), total)
    covariance = model.kern.K(model.X) + np.diag(total)
    expected = multivariate_normal(np.zeros(len(total)), covariance).logpdf(model.Y.ravel())
    assert model.log_likelihood() == pytest.approx(expected, rel=1e-6)
    # Predictive noise adds in fitted units; the pickle keeps the custom likelihood.
    x = np.array([[0.2], [0.7]])
    extra = np.array([0.1, 0.4])
    _, base = gp.predict_transformed(x)
    np.testing.assert_allclose(gp.predict(x, noise=extra)[1], base + extra)
    lo, hi = gp.predict_interval(x, noise=extra)
    np.testing.assert_allclose((hi - lo) / 2, np.sqrt(base + extra))
    restored = load_mfgp(save_mfgp(tmp_path / "gp.pkl", gp))
    np.testing.assert_array_equal(
        restored.predict(x, noise=extra)[1], gp.predict(x, noise=extra)[1]
    )
    with pytest.raises(ValueError, match="noise_list"):
        MultiFidelityGP(2, 1).fit(X, Y, n_restarts=1, noise_list=[None, -np.ones(len(X[1]))])


def test_standalone_binomial_noise(tmp_path):
    from core.surrogates.mfgp_stage import run_mfgp_config
    from tests.test_mfgp_levels import legacy_data

    root = legacy_data(tmp_path / "data")
    levels = [
        {"source": "observed_fraction", "partition": "train", "fidelity": f} for f in ("lf", "hf")
    ]
    common = dict(
        levels=levels,
        evaluation=[{"partition": "validation", "fidelity": "hf"}],
        n_restarts=1,
        output_transform="log",
        data_directory=root,
    )
    out = {}
    for noise in ("learned", "binomial"):
        config = MFGPRunConfig(**common, observation_noise=noise, output_directory=tmp_path / noise)
        out[noise] = run_mfgp_config(config)
    with (
        np.load(out["binomial"] / "fit_arrays.npz") as fit,
        np.load(out["binomial"] / "training_arrays.npz") as raw,
    ):
        events = raw["events_1"].ravel()
        p = fit["Y_1"].ravel()  # smoothed fraction
        np.testing.assert_allclose(fit["noise_1"], events * p * (1 - p) / (events * p + 0.5) ** 2)
    with np.load(out["learned"] / "fit_arrays.npz") as fit:
        assert not any(key.startswith("noise_") for key in fit.files)
    rows = {k: json.loads((v / "metrics.json").read_text())[0] for k, v in out.items()}
    assert rows["binomial"]["noise_rate_out_of_range"] == 0
    assert "observation_noise" not in rows["learned"]
    with (
        np.load(out["learned"] / "evaluation_0.npz") as a,
        np.load(out["binomial"] / "evaluation_0.npz") as b,
    ):
        assert a["sigma"].shape == b["sigma"].shape and not np.allclose(a["sigma"], b["sigma"])
    assert "binomial" in json.loads((out["binomial"] / "model.json").read_text())["noise"]
