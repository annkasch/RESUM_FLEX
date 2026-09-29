import numpy as np
import pytest
import torch

from core.cnp_validation import coverage_counts, evaluate_cnp_validation
from core.surrogate_cnp import build_cnp
from schemas.config import EncoderConfig
from schemas.data_models import InputMode, StandardBatch


def test_coverage_counts_boundaries_and_zero_sigma():
    result = coverage_counts([0, 1, 2, 3, 4], [0] * 5, [0, 1, 1, 1, 1])
    assert [r["within"] for r in result] == [2, 3, 4]
    assert [r["fraction"] for r in result] == [0.4, 0.6, 0.8]
    with pytest.raises(ValueError):
        coverage_counts([0], [0], [-1])
    with pytest.raises(ValueError):
        coverage_counts([np.nan], [0], [1])


def test_validation_determinism_shapes_and_no_training():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = build_cnp(EncoderConfig(latent_dim=8, hidden_dims=[16]), 3, 3)
    rng = np.random.default_rng(1)
    batch = StandardBatch(
        mode=InputMode.FULL,
        theta=rng.normal(size=(4, 3)),
        phi=rng.normal(size=(4, 20, 3)),
        labels=rng.integers(0, 2, size=(4, 20)),
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    a = evaluate_cnp_validation(model, batch, n_context=5, n_mc_samples=20)
    b = evaluate_cnp_validation(model, batch, n_context=5, n_mc_samples=20)
    assert a["summary"] == b["summary"]
    assert a["summary"]["target_events_per_voxel"] == 15
    assert a["observed"].shape == (4,)
    np.testing.assert_allclose(
        a["sigma_total"] ** 2, a["sigma_epistemic"] ** 2 + a["sigma_aleatoric"] ** 2, rtol=1e-5
    )
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.training
    with pytest.raises(ValueError):
        evaluate_cnp_validation(model, batch, n_context=20)
