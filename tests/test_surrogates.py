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


@pytest.mark.parametrize("kind", ["cnp", "mlp", "transformer", "bdt"])
def test_full_experiment_artifacts_and_no_test_access(kind, tmp_path):
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
    config = SurrogateRunConfig(
        model=spec(kind),
        training=training(kind),
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
