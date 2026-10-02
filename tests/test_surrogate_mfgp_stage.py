"""Complete shared neural -> three-fidelity GP artifact workflow."""

import json

import numpy as np
import pytest

from core.surrogates.experiment import run_experiment
from data.pseudo_generator import for_scenario
from schemas.surrogates import SurrogateRunConfig


@pytest.mark.parametrize("transform", ["identity", "log"])
@pytest.mark.parametrize("lf_validation", [True, False])
def test_full_pipeline_and_mfgp_roundtrip(tmp_path, lf_validation, transform):
    pytest.importorskip("GPy")
    source = for_scenario("S1", seed=1)
    root = tmp_path / "data"
    for split in ("train", "validation"):
        for fid in ("lf", "hf"):
            if split == "validation" and fid == "lf" and not lf_validation:
                continue
            batch = source.generate(n_trials=4, n_events=24)
            if transform == "log" and split == "train" and fid == "hf":
                batch.labels[0] = 0  # Exercise zero-count smoothing end to end.
            folder = root / "batches" / split
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(
                folder / f"{fid}.npz",
                mode=batch.mode.value,
                theta=batch.theta,
                phi=batch.phi,
                labels=batch.labels,
            )
    config = SurrogateRunConfig(
        model={"kind": "legacy_cnp", "encoder": {"latent_dim": 8, "hidden_dims": [12]}},
        training={
            "backend": "neural",
            "loss": "theory-truth",
            "n_steps": 2,
            "eval_every": 1,
            "batch_size": 2,
            "n_events": 12,
            "n_context_min": 4,
            "n_context_max": 4,
            "sampling": {"strategy": "class_aware_mixup", "mixup": {"mix_context": True}},
        },
        mfgp={"n_restarts": 1, "n_context": 4, "output_transform": transform},
        data_directory=root,
        output_directory=tmp_path / "run",
        validation_context_events=4,
        lf_validation=lf_validation,
    )
    # Exercise the fine-tuned checkpoint through the complete GP boundary.
    fine_tuned = transform == "log" and lf_validation
    if fine_tuned:
        from schemas.surrogates import LastLayerFineTuning
        config.training.fine_tuning = LastLayerFineTuning(n_steps=2, eval_every=1)
    out = run_experiment(config)
    if fine_tuned:
        from core.surrogates.checkpoints import load_surrogate
        from core.surrogates.pipeline import prepare_surrogate_datasets
        from data.optical_pipeline import load_prepared_batch
        lf = load_prepared_batch(root / "batches/train/lf.npz")
        hf = load_prepared_batch(root / "batches/train/hf.npz")
        def inputs(checkpoint):
            return prepare_surrogate_datasets(
                load_surrogate(out / "checkpoints" / checkpoint), lf, hf,
                n_lf_context=4, n_hf_context=4, seed=config.mfgp.seed)
        expected_inputs, old_inputs = inputs("best"), inputs("pretraining")
        with np.load(out / "mfgp/training_arrays.npz") as arrays:
            for key, expected_array in expected_inputs.items():
                np.testing.assert_array_equal(arrays[key], expected_array)
            assert not np.array_equal(arrays["Y_lf_cnp"], old_inputs["Y_lf_cnp"])

    meta = json.loads((out / "mfgp/model.json").read_text())
    expected = {"train/lf", "train/hf", "validation/hf"}
    if lf_validation:
        expected.add("validation/lf")
    assert set(meta["data"]) == expected
    if not lf_validation:
        assert not (out / "mfgp/lf_validation.npz").exists()
        assert not (out / "lf_means.png").exists()
        assert (out / "training_history.png").exists()
        assert (out / "lf_precision_recall.png").exists()
        manifest = json.loads((out / "experiment.json").read_text())
        assert manifest["selection_split"] is None
        best = json.loads((out / "checkpoints/best/model.json").read_text())
        assert best["metadata"]["step"] == 2
        with np.load(out / "hf_validation_best.npz") as best_arrays, np.load(out / "hf_validation_final.npz") as final_arrays:
            np.testing.assert_array_equal(best_arrays["predicted"], final_arrays["predicted"])
    scores = json.loads((out / "mfgp/metrics.json").read_text())
    for row in scores:
        assert row["mean_bias"] == pytest.approx(row["mean_predicted"] - row["mean_observed"])
        assert row["crps"] >= 0
        assert "weighted_interval_score" in row
        assert "mean_width" in row["interval_metrics"]["2"]
    assert meta["test_data"] == "Not loaded"
    assert (out / "mfgp/hf_coverage.png").exists()
    assert (out / "mfgp/model.pkl").exists()
    with np.load(out / "mfgp/hf_validation.npz") as data:
        assert np.isfinite(data["mean"]).all()
        assert (data["sigma"] >= 0).all()
    with np.load(out / "mfgp/training_arrays.npz") as data:
        assert data["X_hf"].shape[0] == 4

    if transform == "log":
        with (np.load(out / "mfgp/training_arrays.npz") as raw,
              np.load(out / "mfgp/fit_arrays.npz") as fit):
            assert raw["Y_hf_raw"][0, 0] == 0
            assert fit["Y_hf_raw"][0, 0] > 0
            np.testing.assert_allclose(fit["Y_hf_raw"], (raw["Y_hf_raw"] * 20 + 0.5) / 21)
        with np.load(out / "mfgp/hf_validation.npz") as arrays:
            assert (arrays["lower_3"] > 0).all()
            assert (arrays["upper_3"] >= arrays["lower_3"]).all()
