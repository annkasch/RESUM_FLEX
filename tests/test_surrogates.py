"""Cross-model contracts, selection, persistence and downstream integration."""

import json
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from core.surrogate_cnp import split_context_target
from core.surrogates import (
    Episode,
    build_surrogate,
    evaluate_surrogate,
    load_surrogate,
    prepare_surrogate_datasets,
)
from data.pseudo_generator import for_scenario
from schemas.surrogates import NeuralTraining, SurrogateConfig, TreeTraining


def spec(kind):
    if kind == "bdt":
        pytest.importorskip("sklearn")
        return {
            "kind": kind,
            "architecture": {"max_iter": 6, "eval_every": 2, "min_samples_leaf": 2},
        }
    if kind == "mlp":
        return {"kind": kind, "architecture": {"hidden_dims": [12]}}
    if kind == "transformer":
        return {
            "kind": kind,
            "architecture": {
                "architecture": "ft_transformer",
                "token_dim": 8,
                "n_heads": 2,
                "n_layers": 2,
                "feedforward_dim": 12,
            },
        }
    return {"kind": kind, "encoder": {"type": "mlp", "latent_dim": 8, "hidden_dims": [12]}}


def training(kind):
    return (
        TreeTraining()
        if kind == "bdt"
        else NeuralTraining(
            n_steps=4,
            eval_every=2,
            batch_size=3,
            n_events=12,
            n_context_min=2,
            n_context_max=4,
            sampling={"strategy": "positive_quota", "positive_fraction": 0.25},
            weighting={"strategy": "sampling_correction"},
        )
    )


@pytest.mark.parametrize("kind", ["cnp", "mlp", "transformer", "bdt"])
@pytest.mark.parametrize("scenario", [f"S{i}" for i in range(1, 9)])
def test_common_api_all_modes(kind, scenario, tmp_path):
    source = for_scenario(scenario, seed=0)
    train = source.generate(n_trials=5, n_events=24)
    # Guarantee both classes, including for tiny design-only fixtures.
    train.labels[0, 0] = 0
    train.labels[0, 1] = 1
    val = source.generate(n_trials=3, n_events=24, seed=3)
    ctx, tgt = split_context_target(val, 4, seed=7)
    model = build_surrogate(spec(kind), source.dim_theta, source.dim_phi)
    report = model.fit(
        train, training(kind), validation=Episode(ctx, tgt), checkpoints=tmp_path / "run"
    )
    assert report.best_step == min(report.history, key=lambda r: r["voxel_rate_mae"])["step"]
    prediction = model.predict(tgt, context=ctx)
    assert prediction.logits.shape == tgt.labels.shape
    np.testing.assert_allclose(prediction.mean, prediction.probabilities.mean(1))
    altered = tgt.model_copy(update={"labels": 1 - tgt.labels})
    np.testing.assert_array_equal(model.predict(altered, context=ctx).logits, prediction.logits)
    restored = load_surrogate(tmp_path / "run/best")
    np.testing.assert_array_equal(restored.predict(tgt, context=ctx).logits, prediction.logits)
    assert restored.metadata["training"]["backend"] == training(kind).backend
    assert not model.supports_uncertainty
    final = load_surrogate(tmp_path / "run/final")
    assert final.metadata["step"] == (6 if kind == "bdt" else 4)
    if kind == "cnp":
        with pytest.raises(ValueError, match="requires context"):
            model.predict(tgt)
    else:
        np.testing.assert_array_equal(model.predict(tgt).logits, prediction.logits)
    if train.theta is not None:
        data = prepare_surrogate_datasets(
            restored, val, val, n_lf_context=4, n_hf_context=4, seed=6
        )
        assert data["Y_lf_cnp"].shape == (3, 1)
        _, hf_target = split_context_target(val, 4, seed=107)
        np.testing.assert_array_equal(data["Y_hf_raw"][:, 0], hf_target.labels.mean(1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling": {"strategy": "positive_quota", "positive_fraction": 0.05}},
        {"weighting": {"strategy": "sampling_correction"}},
        {"sampling": {"strategy": "natural", "positive_fraction": 0.05}},
        {"n_context_min": 70, "n_context_max": 64},
        {"mixup_alpha": 0.2},
    ],
)
def test_invalid_neural_settings_rejected(kwargs):
    with pytest.raises(ValidationError):
        NeuralTraining(**kwargs)


def test_wrong_backend_and_tree_focal_rejected():
    with pytest.raises(ValidationError):
        SurrogateConfig(model={"kind": "bdt"}, training={"backend": "neural"})
    with pytest.raises(ValidationError):
        TreeTraining(focal_gamma=2)


def test_no_positive_validation_has_explicit_undefined_ap():
    s = for_scenario("S1", seed=0)
    b = s.generate(n_trials=2, n_events=24)
    c, t = split_context_target(b, 4, seed=0)
    t.labels[:] = 0
    m = build_surrogate(spec("mlp"), s.dim_theta, s.dim_phi)
    metrics, arrays = evaluate_surrogate(m, t)
    assert metrics["average_precision"] is None
    assert arrays["recall"].size == 0
    assert np.isfinite(metrics["bernoulli_log_loss"])
    with pytest.raises(ValueError, match="no positives"):
        m.fit(b, training("mlp"), validation=Episode(c, t), selection="average_precision")


def test_optional_backends_do_not_block_neural_models():
    code = """
import builtins
original=builtins.__import__
def blocked(name,*args,**kwargs):
    if name.split('.')[0] in {'sklearn','GPy','emukit'}:
        raise ModuleNotFoundError(name,name=name)
    return original(name,*args,**kwargs)
builtins.__import__=blocked
from core import build_surrogate
m=build_surrogate({'kind':'mlp'},3,6)
assert not m.uses_context
try:build_surrogate({'kind':'bdt'},3,6)
except ImportError as exc:assert '.[bdt]' in str(exc)
else:raise AssertionError('Expected optional-dependency error')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_checkpoint_version_rejected(tmp_path):
    p = tmp_path / "model.json"
    p.write_text(json.dumps({"format": "other", "version": 999}))
    with pytest.raises(ValueError, match="format/version"):
        load_surrogate(tmp_path)


@pytest.mark.parametrize("kind", ["cnp", "mlp", "transformer", "bdt"])
def test_legacy_import_preserves_predictions(kind, tmp_path):
    from core.surrogates import import_legacy_surrogate

    source = for_scenario("S1", seed=0)
    batch = source.generate(n_trials=3, n_events=24)
    batch.labels[0, 0] = 0
    batch.labels[0, 1] = 1
    context, target = split_context_target(batch, 4, seed=2)
    model = build_surrogate(spec(kind), source.dim_theta, source.dim_phi)
    model.fit(batch, training(kind))
    if kind == "bdt":
        from core.surrogate_bdt import save_bdt

        path = tmp_path / "old.joblib"
        save_bdt(path, model.estimator, {})
    else:
        from core.bernoulli_cnp import save_bernoulli_checkpoint
        from core.bernoulli_mlp import save_mlp_checkpoint
        from core.bernoulli_transformer import save_transformer_checkpoint

        save = {
            "cnp": save_bernoulli_checkpoint,
            "mlp": save_mlp_checkpoint,
            "transformer": save_transformer_checkpoint,
        }[kind]
        from schemas.config import EncoderConfig

        encoder = (
            EncoderConfig(type="mlp", latent_dim=1, **model.config.architecture.model_dump())
            if kind == "mlp"
            else getattr(model.config, "encoder", None)
        )
        path = tmp_path / "old.ckpt"
        save(
            path,
            model.module,
            encoder_config=encoder,
            dim_theta=source.dim_theta,
            dim_phi=source.dim_phi,
            metadata={},
        )
    imported = import_legacy_surrogate(path, dim_theta=source.dim_theta, dim_phi=source.dim_phi)
    np.testing.assert_array_equal(
        imported.predict(target, context=context).logits,
        model.predict(target, context=context).logits,
    )
    imported.save(tmp_path / "new")
    np.testing.assert_array_equal(
        load_surrogate(tmp_path / "new").predict(target, context=context).logits,
        model.predict(target, context=context).logits,
    )


@pytest.mark.parametrize(
    "kind,mixup",
    [
        ("cnp", False),
        ("mlp", False),
        ("transformer", False),
        ("bdt", False),
        ("cnp", True),
        ("cnp", "combined"),
    ],
)
def test_full_experiment_artifacts_and_no_test_access(kind, mixup, tmp_path):
    from core.surrogates.experiment import run_experiment
    from schemas.surrogates import SurrogateRunConfig

    source = for_scenario("S1", seed=1)
    root = tmp_path / "data"
    for split, fidelity, seed in [
        ("train", "lf", 1),
        ("validation", "lf", 2),
        ("validation", "hf", 3),
    ]:
        batch = source.generate(n_trials=3, n_events=24, seed=seed)
        batch.labels[:, 0] = 1
        batch.labels[:, 1] = 0
        folder = root / "batches" / split
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{fidelity}.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )
    # No test directory exists: the complete runner must work without it.
    settings = training(kind)
    if mixup:
        settings = NeuralTraining(
            **{
                **settings.model_dump(),
                "sampling": {"strategy": "class_aware_mixup", "mixup": {}},
                "weighting": {"strategy": "none"},
            }
        )
    if mixup == "combined":
        settings = NeuralTraining(
            **{**settings.model_dump(), "objective": {"strategy": "real_plus_mixup"}}
        )
    config = SurrogateRunConfig(
        model=spec(kind),
        training=settings,
        data_directory=root,
        output_directory=tmp_path / "run",
        validation_context_events=4,
    )
    output = run_experiment(config)
    rows = json.loads((output / "metrics.json").read_text())
    assert len(rows) == 6
    assert {r["split"] for r in rows} == {"train", "validation"}
    assert (output / "lf_means.png").exists()
    assert (output / "lf_precision_recall.png").exists()
    manifest = json.loads((output / "checkpoints/best/model.json").read_text())
    assert len(manifest["metadata"]["data"]["train/lf"]["sha256"]) == 64
    with pytest.raises(FileExistsError):
        run_experiment(config)


def test_existing_mfgp_entrypoint_accepts_new_component():
    pytest.importorskip("GPy")
    from core import prepare_mfgp_datasets_from_batches

    s = for_scenario("S1", seed=0)
    b = s.generate(n_trials=3, n_events=24)
    model = build_surrogate(spec("mlp"), s.dim_theta, s.dim_phi)
    old_entry = prepare_mfgp_datasets_from_batches(model, b, b, n_lf_context=4, n_hf_context=4)
    new_entry = prepare_surrogate_datasets(model, b, b, n_lf_context=4, n_hf_context=4)
    for key in old_entry:
        np.testing.assert_array_equal(old_entry[key], new_entry[key])


def test_ap_selection_restores_highest_ap(tmp_path):
    s = for_scenario("S1", seed=7)
    b = s.generate(n_trials=4, n_events=24)
    b.labels[:, :4] = 1
    c, t = split_context_target(b, 4, seed=3)
    model = build_surrogate(spec("mlp"), s.dim_theta, s.dim_phi)
    report = model.fit(b, training("mlp"), validation=Episode(c, t), selection="average_precision")
    expected = max(row["average_precision"] for row in report.history)
    actual, _ = evaluate_surrogate(model, t)
    assert actual["average_precision"] == expected


@pytest.mark.parametrize("kind", ["cnp", "mlp", "transformer"])
@pytest.mark.parametrize("mix_context", [False, True])
def test_mixup_shared_training_and_checkpoint(kind, mix_context, tmp_path, monkeypatch):
    from core.mixup import ClassAwareMixupSource

    source = for_scenario("S1", seed=0)
    batch = source.generate(n_trials=4, n_events=24)
    batch.labels[:, :12] = 0
    batch.labels[:, 12:] = 1
    original = batch.labels.copy()
    context, target = split_context_target(batch, 4, seed=3)
    calls, sums, counts = [], [], []
    original_next = ClassAwareMixupSource.next

    def capture(self, **kwargs):
        ctx, tgt = original_next(self, **kwargs)
        calls.append(ctx.n_events)
        sums.append(float(tgt.labels.sum()))
        counts.append(tgt.labels.size)
        if not mix_context:
            assert np.isin(ctx.labels, [0, 1]).all()
        return ctx, tgt

    monkeypatch.setattr(ClassAwareMixupSource, "next", capture)
    settings = NeuralTraining(
        n_steps=6,
        eval_every=3,
        batch_size=2,
        n_events=12,
        n_context_min=2,
        n_context_max=5,
        sampling={
            "strategy": "class_aware_mixup",
            "mixup": {"alpha": 0.2, "mix_context": mix_context},
        },
    )
    model = build_surrogate(spec(kind), source.dim_theta, source.dim_phi)
    report = model.fit(
        batch, settings, validation=Episode(context, target), checkpoints=tmp_path / "run"
    )
    assert len(set(calls)) > 1
    assert all(2 <= n <= 5 for n in calls)
    assert np.isfinite([r["training_loss"] for r in report.history]).all()
    audit = report.sampling_audit
    assert audit["label_mass"] == sum(sums)
    assert audit["events"] == sum(counts)
    assert audit["mean_target_label"] == sum(sums) / sum(counts)
    assert audit["soft_label_events"] > 0
    assert "positives" not in audit
    np.testing.assert_array_equal(batch.labels, original)
    restored = load_surrogate(tmp_path / "run/best")
    assert restored.metadata["sampling_audit"] == audit
    assert restored.metadata["training"]["sampling"] == settings.sampling.model_dump()
    np.testing.assert_array_equal(
        model.predict(target, context=context).logits,
        restored.predict(target, context=context).logits,
    )


@pytest.mark.parametrize(
    "sampling,weighting",
    [
        ({"strategy": "class_aware_mixup"}, {"strategy": "none"}),
        ({"strategy": "natural", "mixup": {}}, {"strategy": "none"}),
        ({"strategy": "class_aware_mixup", "mixup": {"alpha": 0}}, {"strategy": "none"}),
        (
            {"strategy": "class_aware_mixup", "mixup": {}, "positive_fraction": 0.05},
            {"strategy": "none"},
        ),
        ({"strategy": "class_aware_mixup", "mixup": {}}, {"strategy": "sampling_correction"}),
        ({"strategy": "class_aware_mixup", "mixup": {}}, {"strategy": "class_weights"}),
    ],
)
def test_invalid_mixup_combinations(sampling, weighting):
    with pytest.raises(ValidationError):
        NeuralTraining(sampling=sampling, weighting=weighting)


def test_tree_rejects_mixup():
    with pytest.raises(ValidationError):
        TreeTraining(sampling={"strategy": "class_aware_mixup", "mixup": {}})


@pytest.mark.parametrize("kind", ["cnp", "mlp", "transformer"])
@pytest.mark.parametrize("mix_context", [False, True])
def test_real_plus_mixup_loss_history_counts_and_checkpoint(kind, mix_context, tmp_path):
    s = for_scenario("S1", seed=0)
    b = s.generate(n_trials=4, n_events=24)
    b.labels[:, :12] = 0
    b.labels[:, 12:] = 1
    cfg = NeuralTraining(
        n_steps=3,
        eval_every=1,
        batch_size=2,
        n_events=12,
        n_context_min=4,
        n_context_max=4,
        sampling={"strategy": "class_aware_mixup", "mixup": {"mix_context": mix_context}},
        objective={
            "strategy": "real_plus_mixup",
            "mixup_loss_weight": 0.001,
            "real_target_ratio": 0.5,
        },
    )
    m = build_surrogate(spec(kind), s.dim_theta, s.dim_phi)
    result = m.fit(b, cfg, checkpoints=tmp_path / "fit")
    for row in result.history:
        assert row["training_loss"] == pytest.approx(
            row["training_real_bce"] + 0.001 * row["training_mixup_bce"], rel=1e-6
        )
    audit = result.sampling_audit
    assert audit["real_target_events"] == 24
    assert audit["mixed_target_events"] == 48
    assert audit["total_target_predictions"] == 72
    restored = load_surrogate(tmp_path / "fit/best")
    assert restored.metadata["training"]["objective"] == cfg.objective.model_dump()
    assert restored.metadata["sampling_audit"] == audit


@pytest.mark.parametrize(
    "extra",
    [
        {"sampling": {"strategy": "natural"}},
        {"focal_gamma": 1},
        {"objective": {"strategy": "real_plus_mixup", "real_target_ratio": 0}},
        {"objective": {"strategy": "real_plus_mixup", "mixup_loss_weight": -1}},
        {"objective": {"strategy": "single", "mixup_loss_weight": 0.1}},
    ],
)
def test_invalid_combined_objective(extra):
    args = dict(
        sampling={"strategy": "class_aware_mixup", "mixup": {}},
        objective={"strategy": "real_plus_mixup"},
    )
    args.update(extra)
    with pytest.raises(ValidationError):
        NeuralTraining(**args)


def test_zero_mixup_weight_matches_real_only_optimizer_update():
    import torch

    from core.binary_losses import binary_focal_loss_with_logits
    from core.mixup import ClassAwareMixupSource

    s = for_scenario("S1", seed=0)
    b = s.generate(n_trials=4, n_events=24)
    cfg = NeuralTraining(
        n_steps=1,
        eval_every=1,
        batch_size=2,
        n_events=12,
        n_context_min=4,
        n_context_max=4,
        sampling={"strategy": "class_aware_mixup", "mixup": {}},
        objective={"strategy": "real_plus_mixup", "mixup_loss_weight": 0},
    )
    actual = build_surrogate(spec("cnp"), s.dim_theta, s.dim_phi, seed=11)
    expected = build_surrogate(spec("cnp"), s.dim_theta, s.dim_phi, seed=11)
    sampler = ClassAwareMixupSource(
        b, alpha=0.2, seed=cfg.seed, batch_size=2, n_events=12, n_context=4, mix_context=False
    )
    context, _, real = sampler.next(real_target_ratio=1)
    optimizer = torch.optim.Adam(expected.module.parameters(), lr=cfg.learning_rate)
    loss = binary_focal_loss_with_logits(expected.module(context, real), real.labels, gamma=0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.module.parameters(), cfg.grad_clip)
    optimizer.step()
    actual.fit(b, cfg)
    for a, e in zip(actual.module.parameters(), expected.module.parameters(), strict=True):
        torch.testing.assert_close(a, e)
