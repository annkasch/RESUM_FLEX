"""Complete shared neural -> three-fidelity GP artifact workflow."""

import json

import numpy as np
import pytest

from core.surrogates.experiment import run_experiment
from data.pseudo_generator import for_scenario
from schemas.surrogates import SurrogateRunConfig


def test_full_pipeline_and_mfgp_roundtrip(tmp_path):
    pytest.importorskip("GPy")
    source = for_scenario("S1", seed=1)
    root = tmp_path / "data"
    for split in ("train", "validation"):
        for fid in ("lf", "hf"):
            batch = source.generate(n_trials=4, n_events=24)
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
        mfgp={"n_restarts": 1, "n_context": 4},
        data_directory=root,
        output_directory=tmp_path / "run",
        validation_context_events=4,
    )
    out = run_experiment(config)
    meta = json.loads((out / "mfgp/model.json").read_text())
    assert set(meta["data"]) == {"train/lf", "train/hf", "validation/lf", "validation/hf"}
    assert meta["test_data"] == "Not loaded"
    assert (out / "mfgp/hf_coverage.png").exists()
    assert (out / "mfgp/model.pkl").exists()
    with np.load(out / "mfgp/hf_validation.npz") as data:
        assert np.isfinite(data["mean"]).all()
        assert (data["sigma"] >= 0).all()
    with np.load(out / "mfgp/training_arrays.npz") as data:
        assert data["X_hf"].shape[0] == 4
