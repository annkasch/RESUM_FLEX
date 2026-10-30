import numpy as np
import pytest
import torch

from core.mixup import SoftTargetBatch
from core.surrogate_cnp import CnpOutput, build_cnp, theory_truth_loss
from core.training import train_cnp
from data.batch_source import FixedBatchSource
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig
from schemas.data_models import InputMode, StandardBatch


def batch():
    y = np.array([[0, 0, 0, 0], [0, 0, 1, 1]])
    return StandardBatch(
        mode=InputMode.FULL,
        theta=np.array([[0.0], [1.0]]),
        phi=y[..., None].astype(float),
        labels=y,
    )


def test_soft_target_batch_rejects_invalid_labels():
    original = batch()
    with pytest.raises(ValueError):
        SoftTargetBatch(
            mode=original.mode, theta=original.theta, phi=original.phi,
            labels=np.full((2, 4), np.nan),
        )


def test_soft_cross_entropy_matches_weighted_binary_losses():
    out = CnpOutput(torch.zeros(2, 3), torch.zeros(2, 3))
    weight = 0.3
    soft = theory_truth_loss(out, torch.full((2, 3), weight))
    expected = weight * theory_truth_loss(out, torch.ones(2, 3)) + (1 - weight) * theory_truth_loss(
        out, torch.zeros(2, 3)
    )
    torch.testing.assert_close(soft, expected)


def test_mixup_training_and_objective_guard():
    torch.set_num_threads(1)
    model = build_cnp(EncoderConfig(latent_dim=8, hidden_dims=[16]), 1, 1)
    training = TrainingConfig(
        n_steps=2, batch_size=2, n_events_per_trial=4, eval_every=0, mixup_alpha=0.2
    )
    history = train_cnp(
        model,
        FixedBatchSource(batch()),
        cnp_config=CNPConfig(n_context_min=1, n_context_max=2),
        training_config=training,
    )
    assert np.isfinite(history["loss"]).all()
    with pytest.raises(ValueError, match="theory-truth"):
        train_cnp(
            model,
            FixedBatchSource(batch()),
            cnp_config=CNPConfig(n_context_min=1, n_context_max=2, objective="practice-truth"),
            training_config=training,
        )
