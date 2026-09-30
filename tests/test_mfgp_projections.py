"""Marginalization integrates correlated draws, not pointwise bands or scatter."""

from itertools import product

import numpy as np
import pytest

from core.mfgp_projections import average_grid, marginalize_posterior, projection_grid
from schemas.surrogates import MFGPProjectionConfig


class ConstantCorrelatedGP:
    output_transform = "log"

    def predict_joint_transformed(self, x):
        return np.full(len(x), -5.0), np.full((len(x), len(x)), 0.04)


def test_correlated_lognormal_average_preserves_shared_uncertainty():
    settings = MFGPProjectionConfig(grid_steps=3, n_draws=8192)
    corners = np.array(list(product((0.0, 1.0), repeat=3)))
    arrays, audit = marginalize_posterior(
        ConstantCorrelatedGP(), corners, np.zeros(3), np.ones(3), settings
    )
    # A constant random field has the same distribution after any spatial average.
    for key in ("0", "1", "2", "01", "02", "12"):
        np.testing.assert_allclose(arrays[f"mean_{key}"], np.exp(-5 + 0.04 / 2))
        np.testing.assert_allclose(arrays[f"lower_1_{key}"], np.exp(-5 - 0.2), rtol=0.03)
        np.testing.assert_allclose(arrays[f"upper_1_{key}"], np.exp(-5 + 0.2), rtol=0.03)
        assert np.all(arrays[f"lower_3_{key}"] <= arrays[f"lower_1_{key}"])
        assert np.all(arrays[f"upper_3_{key}"] >= arrays[f"upper_1_{key}"])
    assert audit["valid_grid_points"] == 27
    again, _ = marginalize_posterior(
        ConstantCorrelatedGP(), corners, np.zeros(3), np.ones(3), settings
    )
    np.testing.assert_array_equal(arrays["lower_3_01"], again["lower_3_01"])


def test_domain_mask_and_normalized_averages():
    corners = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    settings = MFGPProjectionConfig(grid_steps=4)
    _, _, points, mask, _ = projection_grid(corners, settings)
    assert np.all(points[mask.ravel()].sum(axis=1) <= 1 + 1e-10)
    assert mask.sum() < 64
    values = np.ones((2, 4, 4, 4)) * 0.03
    values[:, ~mask] = 999  # Outside-domain values must not enter the average.
    result = average_grid(values, mask, (0, 1))
    assert np.isnan(result).any()
    np.testing.assert_allclose(result[np.isfinite(result)], 0.03)


def test_identity_linear_field_and_physical_normalization():
    class LinearGP:
        output_transform = "identity"

        def predict_joint_transformed(self, x):
            # offset=(10,20,30), scale=(2,3,4) must have been applied.
            assert (x >= 0).all() and (x <= 1).all()
            return x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2], np.eye(len(x)) * 1e-10

    offset, scale = np.array([10.0, 20.0, 30.0]), np.array([2.0, 3.0, 4.0])
    corners = np.array(list(product((0.0, 1.0), repeat=3))) * scale + offset
    settings = MFGPProjectionConfig(grid_steps=3, n_draws=2048)
    a, _ = marginalize_posterior(LinearGP(), corners, offset, scale, settings)
    np.testing.assert_allclose(a["mean_0"], (a["axis_0"] - 10) / 2 + 2.5)
    np.testing.assert_allclose(
        a["mean_01"], (a["axis_0"][:, None] - 10) / 2 + 2 * (a["axis_1"][None, :] - 20) / 3 + 1.5
    )


def test_projection_config_requires_explicit_valid_box():
    with pytest.raises(ValueError, match="explicit physical bounds"):
        MFGPProjectionConfig(domain="box")
    with pytest.raises(ValueError, match="finite increasing"):
        MFGPProjectionConfig(bounds=[(0, 1), (1, 0), (0, 1)])
    corners = np.array(list(product((0.0, 1.0), repeat=3)))
    config = MFGPProjectionConfig(domain="box", bounds=[(-1, 2)] * 3, grid_steps=3)
    _, _, _, mask, _ = projection_grid(corners, config)
    assert mask.all()


def test_joint_gp_covariance_excludes_observation_noise():
    pytest.importorskip("GPy")
    from core.surrogate_mfgp import MultiFidelityGP

    x = np.linspace(0, 1, 5)[:, None]
    gp = MultiFidelityGP(2, 1, output_transform="log").fit(
        [x, x], [np.exp(-5 + x), np.exp(-4 + x)], n_restarts=1
    )
    mean, covariance = gp.predict_joint_transformed(x)
    marginal_mean, variance = gp.predict_transformed(x)
    np.testing.assert_allclose(mean, marginal_mean)
    assert np.all(np.diag(covariance) <= variance + 1e-10)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-8


def test_saved_projection_cache_tracks_settings_and_model(tmp_path, monkeypatch):
    pytest.importorskip("GPy")
    import json

    from core.mfgp_projections import ensure_mfgp_projections

    calls = []

    def load(path):
        calls.append(path)
        return ConstantCorrelatedGP()

    def plot(directory):
        for name in ("curves", "curves_observed", "planes", "planes_observed"):
            for ext in ("png", "pdf"):
                (directory / f"{name}.{ext}").write_bytes(b"plot")

    monkeypatch.setattr("core.surrogate_mfgp.load_mfgp", load)
    monkeypatch.setattr("viz.mfgp_projections.plot_mfgp_projections", plot)
    x = np.array(list(product((0.0, 1.0), repeat=3)))
    np.savez(
        tmp_path / "training_arrays.npz", X_lf=x, X_hf=x[:2], Y_hf_raw=np.array([[0.0], [0.01]])
    )
    np.savez(tmp_path / "hf_validation.npz", theta=x[2:], observed=np.arange(6) / 100)
    (tmp_path / "model.pkl").write_bytes(b"model version 1")
    settings = MFGPProjectionConfig(grid_steps=3, n_draws=2048)
    out = ensure_mfgp_projections(tmp_path, settings)
    before = (out / "projections.npz").stat().st_mtime_ns
    ensure_mfgp_projections(tmp_path, settings)
    assert len(calls) == 1
    assert (out / "projections.npz").stat().st_mtime_ns == before
    with np.load(out / "projections.npz") as a:
        np.testing.assert_array_equal(a["observed_validation"], np.arange(6) / 100)
        np.testing.assert_array_equal(a["observed_train"], [0.0, 0.01])
    settings.seed += 1
    ensure_mfgp_projections(tmp_path, settings)
    assert len(calls) == 2
    (tmp_path / "model.pkl").write_bytes(b"model version 2")
    ensure_mfgp_projections(tmp_path, settings)
    assert len(calls) == 3
    np.savez(
        tmp_path / "hf_validation.npz", theta=x[2:], observed=np.arange(6) / 100, target_events=100
    )
    settings.quantity = "observed_fraction"
    predictive_out = ensure_mfgp_projections(tmp_path, settings)
    assert predictive_out.name == "projections_observed_fraction"
    assert json.loads((predictive_out / "metadata.json").read_text())["target_events"] == 100
    assert len(calls) == 4
    assert json.loads((out / "metadata.json").read_text())["signature"]["config"]["seed"] == 43


def test_observed_fraction_matches_binomial_quantiles():
    from scipy.stats import binom

    class ConstantRate:
        output_transform = "identity"

        def predict_joint_transformed(self, x):
            return np.full(len(x), 0.2), np.eye(len(x)) * 1e-20

    settings = MFGPProjectionConfig(
        quantity="observed_fraction", target_events=100, grid_steps=3, n_draws=8192
    )
    corners = np.array(list(product((0.0, 1.0), repeat=3)))
    a, meta = marginalize_posterior(ConstantRate(), corners, np.zeros(3), np.ones(3), settings)
    for key in ("0", "01"):
        np.testing.assert_allclose(a[f"mean_{key}"], 0.2)
        np.testing.assert_allclose(
            a[f"lower_1_{key}"], binom.ppf(0.158655, 100, 0.2) / 100, atol=0.011
        )
        np.testing.assert_allclose(
            a[f"upper_1_{key}"], binom.ppf(0.841345, 100, 0.2) / 100, atol=0.011
        )
        np.testing.assert_allclose(a[f"lower_3_{key}"] * 100, np.round(a[f"lower_3_{key}"] * 100))
    assert meta["target_events"] == 100


def test_observed_projection_retains_spatial_spread():
    class SpatialRate:
        output_transform = "identity"

        def predict_joint_transformed(self, x):
            return np.where(x[:, 2] < 0.5, 0.1, 0.9), np.eye(len(x)) * 1e-20

    corners = np.array(list(product((0.0, 1.0), repeat=3)))
    settings = MFGPProjectionConfig(
        quantity="observed_fraction", target_events=10000, grid_steps=4, n_draws=2048
    )
    a, _ = marginalize_posterior(SpatialRate(), corners, np.zeros(3), np.ones(3), settings)
    np.testing.assert_allclose(a["mean_01"], 0.5)
    assert np.all(a["lower_1_01"] < 0.12)
    assert np.all(a["upper_1_01"] > 0.88)


def test_invalid_binomial_probabilities_are_not_clipped():
    class InvalidGP:
        output_transform = "identity"

        def predict_joint_transformed(self, x):
            return np.full(len(x), 1.1), np.eye(len(x)) * 1e-20

    settings = MFGPProjectionConfig(
        quantity="observed_fraction", target_events=100, grid_steps=3, n_draws=2048
    )
    with pytest.raises(ValueError, match="not silently clipped"):
        marginalize_posterior(
            InvalidGP(),
            np.array(list(product((0.0, 1.0), repeat=3))),
            np.zeros(3),
            np.ones(3),
            settings,
        )
