"""Configurable MFGP levels: schema, standalone runs, inference-file levels, projections."""

import json
from itertools import product

import numpy as np
import pytest
import yaml

from data.pseudo_generator import for_scenario
from schemas.surrogates import (
    MFGPRunConfig,
    MFGPStageConfig,
    SurrogateRunConfig,
    legacy_mfgp_levels,
    load_mfgp_config,
)

TWO_LEVELS = [
    {"source": "model_mean", "partition": "train", "fidelity": "lf"},
    {"source": "observed_fraction", "partition": "train", "fidelity": "hf"},
]


def test_legacy_block_maps_to_three_levels():
    mapped = legacy_mfgp_levels({"max_lf_voxels": 10, "max_hf_voxels": 3}, lf_validation=False)
    config = MFGPStageConfig.model_validate(mapped)
    assert [(lv.source, lv.partition, lv.fidelity, lv.max_voxels) for lv in config.levels] == [
        ("model_mean", "train", "lf", 10),
        ("model_mean", "train", "hf", 3),
        ("observed_fraction", "train", "hf", 3),
    ]
    assert [(e.partition, e.fidelity, e.level) for e in config.evaluation] == [
        ("validation", "hf", None)
    ]
    run = SurrogateRunConfig(
        model={"kind": "mlp"},
        training={"backend": "neural", "n_steps": 2, "eval_every": 1},
        mfgp={"n_restarts": 1},
        data_directory="d",
        output_directory="o",
    )
    assert len(run.mfgp.levels) == 3 and run.mfgp.evaluation[0].level == 0


def test_level_settings_are_validated(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        MFGPStageConfig(levels=TWO_LEVELS[:1])
    with pytest.raises(ValueError, match="same max_voxels"):
        MFGPStageConfig(levels=[{**TWO_LEVELS[1], "max_voxels": 2}, TWO_LEVELS[1]])
    with pytest.raises(ValueError, match="evaluation level"):
        MFGPStageConfig(levels=TWO_LEVELS, evaluation=[{"partition": "v", "level": 2}])
    base = dict(levels=TWO_LEVELS, data_directory="d", output_directory="o")
    with pytest.raises(ValueError, match="need a checkpoint"):
        MFGPRunConfig(**base)
    path = tmp_path / "gp.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "levels": [{**TWO_LEVELS[0], "inference": "runs/inference.npz"}, TWO_LEVELS[1]],
                "data_directory": "data",
                "output_directory": "gp",
            }
        )
    )
    config = load_mfgp_config(path)
    assert config.levels[0].inference == tmp_path / "runs/inference.npz"
    assert config.data_directory == tmp_path / "data" and config.checkpoint is None


def legacy_data(root, seed=1):
    """Prepared .npz batches: train and validation, LF and HF."""
    source = for_scenario("S1", seed=seed)
    for split in ("train", "validation"):
        for fid in ("lf", "hf"):
            batch = source.generate(n_trials=6 if fid == "lf" else 4, n_events=24)
            folder = root / "batches" / split
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(
                folder / f"{fid}.npz",
                mode=batch.mode.value,
                theta=batch.theta,
                phi=batch.phi,
                labels=batch.labels,
            )
    return root


def test_standalone_two_levels_with_checkpoint(tmp_path):
    pytest.importorskip("GPy")
    from core.surrogates import build_surrogate
    from core.surrogates.checkpoints import save_surrogate
    from core.surrogates.mfgp_stage import run_mfgp_config
    from data.optical_pipeline import load_partition

    root = legacy_data(tmp_path / "data")
    lf, _ = load_partition(root, "train", "lf")
    model = build_surrogate(
        {"kind": "mlp", "architecture": {"hidden_dims": [8]}},
        lf.theta.shape[1],
        lf.phi.shape[-1],
        seed=0,
    )
    save_surrogate(tmp_path / "run/checkpoints/best", model)
    config = MFGPRunConfig(
        levels=[{**TWO_LEVELS[0], "max_voxels": 5}, TWO_LEVELS[1]],
        evaluation=[
            {"partition": "validation", "fidelity": "hf"},
            {"partition": "validation", "fidelity": "lf", "level": 0},
        ],
        n_restarts=1,
        output_transform="log",
        data_directory=root,
        checkpoint=tmp_path / "run/checkpoints/best",
        output_directory=tmp_path / "gp",
    )
    out = run_mfgp_config(config)
    meta = json.loads((out / "model.json").read_text())
    assert meta["mode"] == "standalone" and meta["target_level"] == 1
    with np.load(out / "training_arrays.npz") as arrays:
        assert arrays["X_0"].shape[0] == 5 and arrays["X_1"].shape[0] == 4
        np.testing.assert_allclose(
            arrays["Y_0"].ravel(), model.predict(lf.take(arrays["index_0"])).mean
        )
    rows = json.loads((out / "metrics.json").read_text())
    assert [(r["fidelity"], r["level"]) for r in rows] == [("hf", 1), ("lf", 0)]
    for e in range(2):
        assert (out / f"evaluation_{e}_coverage.png").exists()
    assert (out / "training_inputs.png").exists()
    with pytest.raises(FileExistsError):
        run_mfgp_config(config)


def test_inference_file_level_matches_prepared_voxels(tmp_path):
    pytest.importorskip("GPy")
    pytest.importorskip("h5py")
    from core.surrogates.inference import predict_partition
    from core.surrogates.mfgp_stage import run_mfgp_config
    from tests.test_unlabeled_inputs import _checkpoint, prepared

    c, result = prepared(tmp_path)
    checkpoint, model = _checkpoint(tmp_path, "mlp", result.normalization)
    inference = predict_partition(checkpoint, c.output_directory, "gp_unlabeled")
    levels = [
        {"source": "model_mean", "partition": "gp_unlabeled", "inference": str(inference)},
        {"source": "observed_fraction", "partition": "cnp_train", "fidelity": "lf"},
    ]
    config = MFGPRunConfig(
        levels=levels,
        evaluation=[{"partition": "cnp_validation", "fidelity": "lf"}],
        n_restarts=1,
        data_directory=c.output_directory,
        output_directory=tmp_path / "gp",
    )
    out = run_mfgp_config(config)
    with np.load(out / "training_arrays.npz") as arrays, np.load(inference) as saved:
        np.testing.assert_array_equal(arrays["Y_0"].ravel(), saved["mean_probability"])
        np.testing.assert_array_equal(arrays["X_0"], saved["theta"])
    records = json.loads((out / "levels.json").read_text())
    assert records[0]["data"]["inference"] == str(inference) and records[0]["voxels"] == 4
    wrong = config.model_copy(
        update={
            "output_directory": tmp_path / "gp2",
            "levels": [
                config.levels[0].model_copy(update={"partition": "cnp_train"}),
                config.levels[1],
            ],
        }
    )
    with pytest.raises(ValueError, match="holds gp_unlabeled/lf"):
        run_mfgp_config(wrong)


def test_projections_read_flexible_layout(tmp_path, monkeypatch):
    pytest.importorskip("GPy")
    from core.mfgp_projections import ensure_mfgp_projections
    from schemas.projections import ProjectionConfig
    from tests.test_mfgp_projections import ConstantCorrelatedGP

    monkeypatch.setattr("core.surrogate_mfgp.load_mfgp", lambda path: ConstantCorrelatedGP())
    monkeypatch.setattr("viz.mfgp_projections.plot_mfgp_projections", lambda directory: None)
    x = np.array(list(product((0.0, 1.0), repeat=3)))
    np.savez(
        tmp_path / "training_arrays.npz",
        X_0=x,
        Y_0=np.full((8, 1), 0.01),
        X_1=x[:2],
        Y_1=np.array([[0.0], [0.01]]),
    )
    (tmp_path / "levels.json").write_text(
        json.dumps(
            [{"level": 0, "source": "model_mean"}, {"level": 1, "source": "observed_fraction"}]
        )
    )
    (tmp_path / "metrics.json").write_text(json.dumps([{"evaluation": 0, "level": 1}]))
    np.savez(
        tmp_path / "evaluation_0.npz",
        theta=x[2:],
        observed=np.arange(6) / 100,
        target_events=np.full(6, 100.0),
    )
    (tmp_path / "model.pkl").write_bytes(b"model")
    settings = ProjectionConfig(grid_steps=3, n_draws=2048, quantity="observed_fraction")
    out = ensure_mfgp_projections(tmp_path, settings)
    assert json.loads((out / "metadata.json").read_text())["target_events"] == 100
    with np.load(out / "projections.npz") as a:
        np.testing.assert_array_equal(a["observed_train"], [0.0, 0.01])
        np.testing.assert_array_equal(a["observed_validation"], np.arange(6) / 100)
