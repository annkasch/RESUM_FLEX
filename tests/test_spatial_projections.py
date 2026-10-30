"""Adapters share integration, preserve correlations, and separate counting noise."""

from itertools import product

import numpy as np
import pytest
from scipy.special import expit

from core.projection_adapters import (
    BinomialObservation,
    GaussianResponse,
    projection_adapter,
    register_projection_adapter,
)
from core.spatial_projections import project_response
from schemas.projections import ProjectionConfig

CORNERS = np.array(list(product((0.0, 1.0), repeat=3)))


class ConstantLogitAdapter:
    """Latent logit -5 with fully correlated variance 0.04 at every coordinate."""

    def prepare(self, coordinates):
        n = len(coordinates)
        return GaussianResponse(np.full(n, -5.0), np.full((n, n), 0.04), "logit")


def test_logit_joint_marginalization_and_binomial_projection():
    cfg = ProjectionConfig(grid_steps=3, n_draws=8192)
    adapter = ConstantLogitAdapter()
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
    adapter = ConstantLogitAdapter()
    cfg = ProjectionConfig(grid_steps=3, quantity="observed_fraction")
    with pytest.raises(ValueError, match="explicit observation"):
        project_response(adapter, CORNERS, cfg)
    with pytest.raises(ValueError, match="outside"):
        BinomialObservation(100).sample(np.array([[1.1]]), np.random.default_rng(0))
    with pytest.raises(ValueError, match="positive integer"):
        BinomialObservation(1.5)
