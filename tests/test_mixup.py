import numpy as np
import pytest
import torch

from core.mixup import SoftTargetBatch, mixup_targets
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


def test_mixup_alignment_zero_files_and_binary_contract():
    original = batch()
    mixed = mixup_targets(original, alpha=0.2, rng=np.random.default_rng(4))
    np.testing.assert_allclose(mixed.phi[..., 0], mixed.labels)
    # Independent pair weights allow the realized batch mean to fluctuate.
    assert not np.isclose(mixed.labels[1].mean(), original.labels[1].mean())
    np.testing.assert_array_equal(mixed.theta, original.theta)
    np.testing.assert_array_equal(mixed.labels[0], 0)
    assert np.any((mixed.labels > 0) & (mixed.labels < 1))
    np.testing.assert_array_equal(original.labels, [[0, 0, 0, 0], [0, 0, 1, 1]])
    with pytest.raises(ValueError, match="binary"):
        StandardBatch(mode=mixed.mode, theta=mixed.theta, phi=mixed.phi, labels=mixed.labels)
    with pytest.raises(ValueError):
        SoftTargetBatch(
            mode=mixed.mode, theta=mixed.theta, phi=mixed.phi, labels=np.full((2, 4), np.nan)
        )
    again = mixup_targets(original, alpha=0.2, rng=np.random.default_rng(4))
    np.testing.assert_array_equal(mixed.labels, again.labels)


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
