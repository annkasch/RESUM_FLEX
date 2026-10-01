"""Verify head-only fitting, natural targets, persistence and held-out testing."""

import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from core.surrogate_cnp import split_context_target
from core.surrogates import Episode, build_surrogate, load_surrogate, prepare_surrogate_datasets
from core.surrogates.experiment import run_experiment
from data.pseudo_generator import for_scenario
from schemas.surrogates import NeuralTraining, SurrogateRunConfig
from tests.test_surrogates import spec


def settings(kind):
    return NeuralTraining(
        n_steps=3,
        eval_every=1,
        n_events=12,
        batch_size=3,
        n_context_min=4,
        n_context_max=4,
        loss="theory-truth" if kind == "legacy_cnp" else "bernoulli",
        sampling={"strategy": "class_aware_mixup", "mixup": {"mix_context": True}},
        fine_tuning={"n_steps": 3, "eval_every": 1, "learning_rate": 0.01, "n_events": 16},
    )


@pytest.mark.parametrize("kind", ["cnp", "legacy_cnp", "mlp", "transformer"])
def test_only_final_layer_changes_with_real_targets(kind, tmp_path):
    source = for_scenario("S1", seed=3)
    batch = source.generate(n_trials=4, n_events=24)
    batch.labels[:] = 0
    batch.labels[:, 0] = 1
    ctx, tgt = split_context_target(batch, 4, seed=17)
    model = build_surrogate(spec(kind), source.dim_theta, source.dim_phi, seed=1)
    report = model.fit(batch, settings(kind), validation=Episode(ctx, tgt), checkpoints=tmp_path)
    pre = load_surrogate(tmp_path / "pretraining")
    names = set(model.metadata["fine_tuning"]["trainable_parameters"])
    changed = []
    for name, p in model.module.named_parameters():
        before = dict(pre.module.named_parameters())[name]
        if name not in names:
            torch.testing.assert_close(p, before, rtol=0, atol=0)
        elif not torch.equal(p, before):
            changed.append(name)
    assert changed
    assert all(p.requires_grad for p in model.module.parameters())
    assert report.best_step >= 4
    assert {r["stage"] for r in report.history} == {"pretraining", "fine_tuning"}
    audit = report.sampling_audit["fine_tuning"]
    assert audit["strategy"] == "natural"
    assert audit["requested_positive_fraction"] is None
    fine = model.metadata["fine_tuning"]["settings"]
    assert fine["focal_gamma"] == 0 and fine["weighting"]["strategy"] == "none"
    assert fine["sampling"]["strategy"] == "natural"
    restored = load_surrogate(tmp_path / "best")
    np.testing.assert_array_equal(
        model.predict(tgt, context=ctx).logits, restored.predict(tgt, context=ctx).logits
    )
    assert restored.metadata["training"]["fine_tuning"]["n_steps"] == 3
    arrays = prepare_surrogate_datasets(restored, batch, batch, n_lf_context=4, n_hf_context=4)
    assert arrays["Y_lf_cnp"].shape == (4, 1)
    assert load_surrogate(tmp_path / "final").metadata["step"] == 6


def test_frozen_dropout_features_are_deterministic(tmp_path):
    source = for_scenario("S1", seed=0)
    batch = source.generate(n_trials=3, n_events=24)
    cfg = spec("cnp")
    cfg["encoder"]["dropout"] = 0.5
    model = build_surrogate(cfg, source.dim_theta, source.dim_phi)
    modes = []
    model.module.encoder.register_forward_pre_hook(
        lambda module, args: modes.append(module.training)
    )
    model.fit(batch, settings("cnp"), checkpoints=tmp_path)
    # Three pretraining calls each encode context and target; all final six calls
    # (three fine-tuning updates) use deterministic frozen features.
    assert any(modes[:-6]) and modes[-6:] == [False] * 6


def test_test_dataset_is_loaded_after_fit_only(tmp_path, monkeypatch):
    from core.surrogates.models import NeuralSurrogate
    from data import optical_pipeline

    source = for_scenario("S1", seed=0)
    root = tmp_path / "data"
    for split, shift in [("train", 0), ("validation", 10), ("test", 20)]:
        batch = source.generate(n_trials=3, n_events=24)
        batch.theta += shift
        folder = root / "batches" / split
        folder.mkdir(parents=True)
        np.savez(
            folder / "lf.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )
    fitted = False
    original_fit, original_load = NeuralSurrogate.fit, optical_pipeline.load_prepared_batch

    def fit(*args, **kwargs):
        nonlocal fitted
        result = original_fit(*args, **kwargs)
        fitted = True
        return result

    def load(path):
        if "/test/" in str(path):
            assert fitted
        return original_load(path)

    monkeypatch.setattr(NeuralSurrogate, "fit", fit)
    monkeypatch.setattr(optical_pipeline, "load_prepared_batch", load)
    out = run_experiment(
        SurrogateRunConfig(
            model=spec("cnp"),
            training=settings("cnp"),
            data_directory=root,
            output_directory=tmp_path / "run",
            validation_context_events=4,
            hf_validation=False,
            test_fidelities=["lf"],
        )
    )
    rows = json.loads((out / "metrics.json").read_text())
    assert len(rows) == 9
    assert {r["checkpoint"] for r in rows} == {"pretraining", "best", "final"}
    assert (out / "lf_test_fine_tuning_means.png").exists()
    assert (out / "lf_test_fine_tuning_pr.png").exists()
    assert not json.loads((out / "test_evaluation.json").read_text())["used_for_selection"]
    with np.load(out / "lf_test_pretraining.npz") as pre, np.load(out / "lf_test_best.npz") as post:
        np.testing.assert_array_equal(pre["labels"], post["labels"])


def test_fine_tuning_invalid_sizes_rejected():
    with pytest.raises(ValidationError, match="Fine-tuning events"):
        NeuralTraining(fine_tuning={"n_events": 10})
    source = for_scenario("S1", seed=0)
    batch = source.generate(n_trials=3, n_events=24)
    model = build_surrogate(spec("cnp"), source.dim_theta, source.dim_phi)
    cfg = settings("cnp").model_copy(
        update={"fine_tuning": settings("cnp").fine_tuning.model_copy(update={"n_events": 30})}
    )
    with pytest.raises(ValueError, match="Fine-tuning n_events exceeds"):
        model.fit(batch, cfg)
