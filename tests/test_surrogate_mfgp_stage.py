"""Complete shared neural -> multi-fidelity GP artifact workflow (combined mode)."""

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
        mfgp={
            "n_restarts": 1,
            "n_context": 4,
            "output_transform": transform,
            "levels": [
            {"source": "model_mean", "partition": "train", "fidelity": "lf"},
            {"source": "model_mean", "partition": "train", "fidelity": "hf"},
            {"source": "observed_fraction", "partition": "train", "fidelity": "hf"},
        ],
            "evaluation": (
                [{"partition": "validation", "fidelity": "lf", "level": 0}] if lf_validation
                else []
            ) + [{"partition": "validation", "fidelity": "hf"}],
        },
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
        from core.mfgp_pipeline import model_mean
        from core.surrogates.checkpoints import load_surrogate
        from data.optical_pipeline import load_prepared_batch
        lf = load_prepared_batch(root / "batches/train/lf.npz")
        hf = load_prepared_batch(root / "batches/train/hf.npz")
        seed = config.mfgp.seed

        def means(checkpoint, batch, stream):
            model = load_surrogate(out / "checkpoints" / checkpoint)
            # Level streams 0 (train/lf) and 1 (train/hf) split with seed + 1 + 100 stream.
            return model_mean(model, batch, n_context=4, seed=seed + 1 + 100 * stream)

        with np.load(out / "mfgp/training_arrays.npz") as arrays:
            np.testing.assert_array_equal(arrays["Y_0"].ravel(), means("best", lf, 0))
            np.testing.assert_array_equal(arrays["Y_1"].ravel(), means("best", hf, 1))
            assert not np.array_equal(arrays["Y_0"].ravel(), means("pretraining", lf, 0))

    meta = json.loads((out / "mfgp/model.json").read_text())
    assert meta["mode"] == "combined" and meta["target_level"] == 2
    assert meta["levels"] == [
        "model_mean train/lf", "model_mean train/hf", "observed_fraction train/hf"
    ]
    levels = json.loads((out / "mfgp/levels.json").read_text())
    assert [r["voxels"] for r in levels] == [4, 4, 4]
    scores = json.loads((out / "mfgp/metrics.json").read_text())
    expected = [("validation", "hf", 2)]
    if lf_validation:
        expected.insert(0, ("validation", "lf", 0))
    assert [(r["partition"], r["fidelity"], r["level"]) for r in scores] == expected
    if not lf_validation:
        assert not (out / "lf_means.png").exists()
        assert (out / "training_history.png").exists()
        assert (out / "lf_precision_recall.png").exists()
        manifest = json.loads((out / "experiment.json").read_text())
        assert manifest["selection_split"] is None
        best = json.loads((out / "checkpoints/best/model.json").read_text())
        assert best["metadata"]["step"] == 2
        with np.load(out / "hf_validation_best.npz") as best_arrays, np.load(out / "hf_validation_final.npz") as final_arrays:
            np.testing.assert_array_equal(best_arrays["predicted"], final_arrays["predicted"])
    for row in scores:
        assert row["mean_bias"] == pytest.approx(row["mean_predicted"] - row["mean_observed"])
        assert row["crps"] >= 0
        assert "weighted_interval_score" in row
        assert "mean_width" in row["interval_metrics"]["2"]
    hf = len(scores) - 1
    assert (out / f"mfgp/evaluation_{hf}_coverage.png").exists()
    assert (out / "mfgp/training_inputs.png").exists()
    assert (out / "mfgp/model.pkl").exists()
    with np.load(out / f"mfgp/evaluation_{hf}.npz") as data:
        assert np.isfinite(data["mean"]).all()
        assert (data["sigma"] >= 0).all()
        assert data["model_mean"].shape == data["mean"].shape
    with np.load(out / "mfgp/training_arrays.npz") as data:
        assert data["X_2"].shape[0] == 4
        # Observed fractions use every event of the voxel.
        np.testing.assert_allclose(data["events_2"], 24)

    if transform == "log":
        with (np.load(out / "mfgp/training_arrays.npz") as raw,
              np.load(out / "mfgp/fit_arrays.npz") as fit):
            assert raw["Y_2"][0, 0] == 0
            assert fit["Y_2"][0, 0] > 0
            np.testing.assert_allclose(fit["Y_2"], (raw["Y_2"] * 24 + 0.5) / 25)
        with np.load(out / f"mfgp/evaluation_{hf}.npz") as arrays:
            assert (arrays["lower_3"] > 0).all()
            assert (arrays["upper_3"] >= arrays["lower_3"]).all()
