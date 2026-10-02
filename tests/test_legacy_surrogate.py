"""Legacy two-output training and shared API contracts."""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from core.mixup import ClassAwareMixupSource
from core.surrogate_cnp import (
    CnpOutput,
    cnp_loss,
    resum_binary_logits,
    resum_binary_moments,
    split_context_target,
)
from core.surrogates import Episode, build_surrogate, evaluate_surrogate, load_surrogate
from data.pseudo_generator import for_scenario
from schemas.surrogates import NeuralTraining, SurrogateConfig


def legacy_spec():
    return {"kind": "legacy_cnp", "encoder": {"latent_dim": 8, "hidden_dims": [12]}}


def settings(loss, mix_context=False):
    return NeuralTraining(
        n_steps=2,
        eval_every=1,
        batch_size=2,
        n_events=12,
        n_context_min=4,
        n_context_max=4,
        loss=loss,
        sampling={"strategy": "class_aware_mixup", "mixup": {"mix_context": mix_context}},
        objective={"strategy": "real_plus_mixup", "mixup_loss_weight": 0.01},
    )


@pytest.mark.parametrize("scenario", [f"S{i}" for i in range(1, 9)])
@pytest.mark.parametrize("loss", ["theory-truth", "practice-truth"])
@pytest.mark.parametrize("mix_context", [False, True])
def test_legacy_combined_all_modes(scenario, loss, mix_context, tmp_path):
    src = for_scenario(scenario, seed=0)
    batch = src.generate(n_trials=3, n_events=24)
    batch.labels[:, :12] = 0
    batch.labels[:, 12:] = 1
    ctx, tgt = split_context_target(batch, 4, seed=2)
    model = build_surrogate(legacy_spec(), src.dim_theta, src.dim_phi)
    before = model.module.decoder.net[-1].weight.detach().clone()
    result = model.fit(
        batch,
        settings(loss, mix_context),
        validation=Episode(ctx, tgt),
        checkpoints=tmp_path / "fit",
    )
    suffix = "nll" if loss == "practice-truth" else "bce"
    for row in result.history:
        assert row["training_loss"] == pytest.approx(
            row[f"training_real_{suffix}"] + 0.01 * row[f"training_mixup_{suffix}"], rel=1e-6
        )
    after = model.module.decoder.net[-1].weight.detach()
    assert torch.any(after[0] != before[0]) and torch.any(after[1] != before[1])
    with torch.no_grad():
        mean, scale = resum_binary_moments(model.module(ctx, tgt))
    prediction = model.predict(tgt, context=ctx)
    np.testing.assert_allclose(prediction.probabilities, mean.numpy(), rtol=1e-6)
    np.testing.assert_array_equal(prediction.legacy_scale, scale.numpy())
    loaded = load_surrogate(tmp_path / "fit/best")
    restored = loaded.predict(tgt, context=ctx)
    np.testing.assert_array_equal(prediction.logits, restored.logits)
    np.testing.assert_array_equal(prediction.legacy_scale, restored.legacy_scale)
    assert loaded.metadata["training"]["loss"] == loss
    _, arrays = evaluate_surrogate(loaded, tgt, context=ctx)
    np.testing.assert_array_equal(arrays["legacy_scale"], prediction.legacy_scale)
    with pytest.raises(ValueError, match="requires context"):
        model.predict(tgt)


@pytest.mark.parametrize("loss", ["theory-truth", "practice-truth"])
def test_combined_update_matches_original_legacy_loss(loss):
    src = for_scenario("S1", seed=0)
    batch = src.generate(n_trials=3, n_events=24)
    cfg = settings(loss).model_copy(update={"n_steps": 1})
    actual = build_surrogate(legacy_spec(), src.dim_theta, src.dim_phi, seed=4)
    reference = build_surrogate(legacy_spec(), src.dim_theta, src.dim_phi, seed=4)
    sampler = ClassAwareMixupSource(
        batch, alpha=0.2, seed=0, batch_size=2, n_events=12, n_context=4, mix_context=False
    )
    ctx, mixed, real = sampler.next(real_target_ratio=1)
    optimizer = torch.optim.Adam(reference.module.parameters(), lr=cfg.learning_rate)
    real_loss = cnp_loss(
        reference.module(ctx, real), torch.tensor(real.labels, dtype=torch.float32), objective=loss
    )
    mixed_loss = cnp_loss(
        reference.module(ctx, mixed),
        torch.tensor(mixed.labels, dtype=torch.float32),
        objective=loss,
    )
    (real_loss + 0.01 * mixed_loss).backward()
    torch.nn.utils.clip_grad_norm_(reference.module.parameters(), cfg.grad_clip)
    optimizer.step()
    actual.fit(batch, cfg)
    for a, b in zip(actual.module.parameters(), reference.module.parameters(), strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    "kind,loss", [("legacy_cnp", "bernoulli"), ("cnp", "practice-truth"), ("mlp", "theory-truth")]
)
def test_legacy_loss_requires_matching_model(kind, loss):
    with pytest.raises(ValidationError):
        SurrogateConfig(model={"kind": kind}, training={"backend": "neural", "loss": loss})


@pytest.mark.parametrize("loss", ["theory-truth", "practice-truth"])
def test_legacy_natural_single_objective(loss):
    src = for_scenario("S1", seed=0)
    batch = src.generate(n_trials=3, n_events=24)
    cfg = NeuralTraining(
        n_steps=1,
        eval_every=1,
        batch_size=2,
        n_events=12,
        n_context_min=4,
        n_context_max=4,
        loss=loss,
    )
    model = build_surrogate(legacy_spec(), src.dim_theta, src.dim_phi)
    result = model.fit(batch, cfg)
    assert np.isfinite(result.history[0]["training_loss"])


@pytest.mark.parametrize("loss", ["theory-truth", "practice-truth"])
def test_legacy_full_runner(loss, tmp_path):
    import json

    from core.surrogates.experiment import run_experiment
    from schemas.surrogates import SurrogateRunConfig

    source = for_scenario("S1", seed=1)
    root = tmp_path / "data"
    for split, fidelity in [("train", "lf"), ("validation", "lf"), ("validation", "hf")]:
        batch = source.generate(n_trials=3, n_events=24)
        batch.labels[:, 0] = 1
        folder = root / "batches" / split
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{fidelity}.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )
    output = run_experiment(
        SurrogateRunConfig(
            model=legacy_spec(),
            training=settings(loss),
            data_directory=root,
            output_directory=tmp_path / "run",
            validation_context_events=4,
        )
    )
    assert (output / "lf_means.png").exists()
    assert (output / "lf_precision_recall.png").exists()
    with np.load(output / "lf_validation_best.npz") as arrays:
        assert arrays["legacy_scale"].shape == arrays["labels"].shape
        assert np.isin(arrays["labels"], [0, 1]).all()
    manifest = json.loads((output / "checkpoints/best/model.json").read_text())
    assert manifest["model"]["kind"] == "legacy_cnp"
    assert manifest["metadata"]["training"]["loss"] == loss


@pytest.mark.parametrize("loss", ["theory-truth", "practice-truth"])
def test_legacy_loss_applies_importance_weights(loss):
    torch.manual_seed(0)
    out = CnpOutput(mu_logit=torch.randn(2, 6), log_sigma=torch.randn(2, 6))
    labels = torch.tensor([[1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=torch.float32)
    weights = torch.rand(2, 6) + 0.5
    if loss == "theory-truth":
        per_event = torch.nn.functional.binary_cross_entropy_with_logits(
            resum_binary_logits(out), labels, reduction="none"
        )
    else:
        mean, scale = resum_binary_moments(out)
        per_event = -torch.distributions.Normal(mean, scale).log_prob(labels)
    weighted = cnp_loss(out, labels, objective=loss, weights=weights)
    torch.testing.assert_close(weighted, (weights * per_event).mean())
    torch.testing.assert_close(
        cnp_loss(out, labels, objective=loss, weights=torch.ones(2, 6)),
        cnp_loss(out, labels, objective=loss),
    )


@pytest.mark.parametrize("loss", ["theory-truth", "practice-truth"])
def test_legacy_positive_quota_with_sampling_correction(loss, tmp_path):
    src = for_scenario("S1", seed=0)
    batch = src.generate(n_trials=3, n_events=40)
    batch.labels[:] = 0
    batch.labels[:, :3] = 1
    ctx, tgt = split_context_target(batch, 4, seed=2)
    training = NeuralTraining(
        n_steps=3,
        eval_every=1,
        batch_size=2,
        n_events=20,
        n_context_min=4,
        n_context_max=6,
        loss=loss,
        sampling={"strategy": "positive_quota", "positive_fraction": 0.25},
        weighting={"strategy": "sampling_correction"},
    )
    model = build_surrogate(legacy_spec(), src.dim_theta, src.dim_phi)
    result = model.fit(batch, training, validation=Episode(ctx, tgt), checkpoints=tmp_path)
    audit = result.sampling_audit
    assert audit["strategy"] == "positive_quota"
    assert audit["requested_positive_fraction"] == 0.25
    # Quota oversamples far above the natural 3/40 positive rate.
    assert audit["positive_fraction"] > 0.2
    assert all(np.isfinite(row["training_loss"]) for row in result.history)
    restored = load_surrogate(tmp_path / "best")
    assert restored.metadata["training"]["weighting"]["strategy"] == "sampling_correction"


def test_legacy_loss_still_rejects_focal_gamma():
    with pytest.raises(ValidationError):
        NeuralTraining(loss="theory-truth", focal_gamma=2)
