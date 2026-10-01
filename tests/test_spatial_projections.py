"""Adapters share integration, preserve correlations, and separate counting noise."""

from itertools import product

import numpy as np
import pytest
from scipy.special import expit

from core.projection_adapters import (
    BinomialGPProjectionAdapter,
    BinomialObservation,
    GaussianResponse,
    projection_adapter,
    register_projection_adapter,
)
from core.spatial_projections import project_response
from schemas.projections import ProjectionConfig

CORNERS = np.array(list(product((0.0, 1.0), repeat=3)))


class ConstantLogit:
    def predict_latent(self, coordinates, full_cov=False):
        assert full_cov
        return np.full(len(coordinates), -5.0), np.full((len(coordinates), len(coordinates)), 0.04)


def test_logit_joint_marginalization_and_binomial_projection():
    cfg = ProjectionConfig(grid_steps=3, n_draws=8192)
    adapter = BinomialGPProjectionAdapter(ConstantLogit())
    latent, meta = project_response(adapter, CORNERS, cfg)
    for key in ("0", "1", "2", "01", "02", "12"):
        np.testing.assert_allclose(latent[f"lower_1_{key}"], expit(-5 - 0.2), rtol=0.03)
        np.testing.assert_allclose(latent[f"upper_1_{key}"], expit(-5 + 0.2), rtol=0.03)
    assert meta["transform"] == "logit"
    cfg = cfg.model_copy(update={"quantity": "observed_fraction", "target_events": 1500})
    predictive, metadata = project_response(
        adapter, CORNERS, cfg, observation_model=BinomialObservation(1500)
    )
    assert metadata["target_events"] == 1500
    np.testing.assert_array_equal(latent["mean_0"], predictive["mean_0"])
    np.testing.assert_allclose(
        predictive["lower_1_0"] * 1500, np.round(predictive["lower_1_0"] * 1500)
    )
    assert np.mean(predictive["upper_1_0"] - predictive["lower_1_0"]) > np.mean(
        latent["upper_1_0"] - latent["lower_1_0"]
    )


def test_new_backend_requires_no_engine_changes():
    class CustomAdapter:
        def __init__(self, model):
            self.model = model

        def prepare(self, coordinates):
            return GaussianResponse(
                np.full(len(coordinates), self.model), np.eye(len(coordinates)) * 1e-20, "identity"
            )

    register_projection_adapter("test_constant", CustomAdapter)
    adapter = projection_adapter("test_constant", 0.2)
    arrays, _ = project_response(adapter, CORNERS, ProjectionConfig(grid_steps=3, n_draws=2048))
    np.testing.assert_allclose(arrays["mean_01"], 0.2)
    with pytest.raises(ValueError, match="already registered"):
        register_projection_adapter("test_constant", CustomAdapter)


def test_noise_must_be_explicit_and_probabilities_valid():
    adapter = BinomialGPProjectionAdapter(ConstantLogit())
    cfg = ProjectionConfig(grid_steps=3, quantity="observed_fraction")
    with pytest.raises(ValueError, match="explicit observation"):
        project_response(adapter, CORNERS, cfg)
    with pytest.raises(ValueError, match="outside"):
        BinomialObservation(100).sample(np.array([[1.1]]), np.random.default_rng(0))
    with pytest.raises(ValueError, match="positive integer"):
        BinomialObservation(1.5)


def test_saved_count_projection_cache_and_mixed_budgets(tmp_path, monkeypatch):
    pytest.importorskip("GPy")
    from core.count_gp_projections import ensure_count_gp_projections

    calls = []

    def load(path):
        calls.append(path)
        return ConstantLogit()

    def plot(path):
        for name in ("curves", "curves_observed", "planes", "planes_observed"):
            for ext in ("png", "pdf"):
                (path / f"{name}.{ext}").touch()

    monkeypatch.setattr("core.binomial_gp.BinomialGP.load", load)
    monkeypatch.setattr("viz.spatial_projections.plot_spatial_projections", plot)
    (tmp_path / "model.pkl").write_bytes(b"model")
    for split in ("train", "validation"):
        np.savez(
            tmp_path / f"{split}_counts.npz",
            theta=CORNERS,
            hits=np.ones(8),
            trials=np.full(8, 5000),
            source_group=np.array(["hf"] * 8),
        )
    cfg = ProjectionConfig(grid_steps=3, n_draws=2048, quantity="observed_fraction")
    out = ensure_count_gp_projections(tmp_path, cfg)
    ensure_count_gp_projections(tmp_path, cfg)
    assert len(calls) == 1
    assert (out / "projections.npz").exists()
    np.savez(
        tmp_path / "validation_counts.npz",
        theta=CORNERS,
        hits=np.ones(8),
        trials=np.arange(8) + 5000,
        source_group=np.array(["hf"] * 8),
    )
    with pytest.raises(ValueError, match="Mixed validation budgets"):
        ensure_count_gp_projections(tmp_path, cfg)
    ensure_count_gp_projections(tmp_path, cfg.model_copy(update={"target_events": 5000}))
    assert len(calls) == 2
