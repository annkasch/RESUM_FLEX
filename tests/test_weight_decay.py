"""Weight decay of neural training: AdamW on weight matrices only, 0 = plain Adam."""

import pytest
import torch
from pydantic import ValidationError

from core.surrogates import build_surrogate
from core.surrogates.training import build_optimizer
from schemas.surrogates import NeuralTraining, StreamTraining


def _model():
    return build_surrogate({"kind": "mlp", "architecture": {"hidden_dims": [8, 8]}}, 2, 3)


def test_weight_decay_defaults_to_plain_adam_and_rejects_negative():
    assert NeuralTraining().weight_decay == 0 and StreamTraining().weight_decay == 0
    optimizer = build_optimizer(_model().module.parameters(), learning_rate=1e-3)
    assert type(optimizer) is torch.optim.Adam
    for config in (NeuralTraining, StreamTraining):
        with pytest.raises(ValidationError):
            config(weight_decay=-0.1)


def test_weight_decay_shrinks_weight_matrices_only():
    module = _model().module
    named = dict(module.named_parameters())
    assert any(p.ndim < 2 for p in named.values())  # biases and null embeddings exist
    before = {name: p.detach().clone() for name, p in named.items()}
    optimizer = build_optimizer(module.parameters(), learning_rate=0.1, weight_decay=0.5)
    assert type(optimizer) is torch.optim.AdamW
    for p in module.parameters():
        p.grad = torch.zeros_like(p)  # no loss gradient: only the decay moves parameters
    optimizer.step()
    for name, p in named.items():
        expected = before[name] * (1 - 0.1 * 0.5) if p.ndim >= 2 else before[name]
        torch.testing.assert_close(p.detach(), expected, msg=name)


def test_design_point_fit_with_weight_decay_moves_weights_toward_zero():
    from core.surrogate_cnp import split_context_target
    from core.surrogates import Episode
    from data.pseudo_generator import for_scenario

    source = for_scenario("S1", seed=7)
    batch = source.generate(n_trials=4, n_events=24)
    batch.labels[:, :4] = 1
    context, target = split_context_target(batch, 4, seed=3)
    norms = {}
    for decay in (0.0, 5.0):
        model = build_surrogate(
            {"kind": "mlp", "architecture": {"hidden_dims": [12]}},
            source.dim_theta,
            source.dim_phi,
        )
        training = NeuralTraining(
            n_steps=20, eval_every=10, batch_size=3, n_events=12, learning_rate=0.01,
            weight_decay=decay,
        )  # fmt: skip
        model.fit(batch, training, validation=Episode(context, target))
        norms[decay] = sum(
            float(p.detach().square().sum()) for p in model.module.parameters() if p.ndim >= 2
        )
    assert norms[5.0] < norms[0.0]
