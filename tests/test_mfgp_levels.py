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
    load_mfgp_config,
)

TWO_LEVELS = [
    {"source": "model_mean", "partition": "train", "fidelity": "lf"},
    {"source": "observed_fraction", "partition": "train", "fidelity": "hf"},
]


def test_mfgp_blocks_need_explicit_levels():
    base = dict(
        model={"kind": "mlp"},
        training={"backend": "neural", "n_steps": 2, "eval_every": 1},
        data_directory="d",
        output_directory="o",
    )
    with pytest.raises(ValueError, match="levels"):
        SurrogateRunConfig(**base, mfgp={"n_restarts": 1})
    with pytest.raises(ValueError, match="max_lf_voxels"):
        SurrogateRunConfig(**base, mfgp={"levels": TWO_LEVELS, "max_lf_voxels": 3})
    run = SurrogateRunConfig(**base, mfgp={"levels": TWO_LEVELS})
    assert [lv.source for lv in run.mfgp.levels] == ["model_mean", "observed_fraction"]


def test_level_settings_are_validated(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        MFGPStageConfig(levels=TWO_LEVELS[:1])
    with pytest.raises(ValueError, match="same max_voxels"):
        MFGPStageConfig(levels=[{**TWO_LEVELS[1], "max_voxels": 2}, TWO_LEVELS[1]])
    with pytest.raises(ValueError, match="blinded"):
        MFGPStageConfig(levels=TWO_LEVELS, evaluation=[{"partition": "gp_test"}])
    unblinded = MFGPStageConfig(
        levels=TWO_LEVELS, evaluation=[{"partition": "gp_test"}], unblind_test=True
    )
    assert unblinded.evaluation[0].partition == "gp_test"
    with pytest.raises(ValueError, match="can't be a GP training level"):
        MFGPStageConfig(levels=[TWO_LEVELS[0], {**TWO_LEVELS[1], "partition": "cnp_test"}],
                        unblind_test=True)
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
    from core.surrogates.datasets import NpzBatches
    from core.surrogates.mfgp_stage import run_mfgp_config

    root = legacy_data(tmp_path / "data")
    lf, _ = NpzBatches(root).load("train", "lf")
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


@pytest.mark.parametrize("show", [True, False])
def test_coverage_plot_can_hide_predicted_line(tmp_path, monkeypatch, show):
    import matplotlib.axes

    from viz.dispatch import plot_coverage_test

    labels = []
    original = matplotlib.axes.Axes.plot

    def plot(self, *args, label=None, **kwargs):
        labels.append(label)
        return original(self, *args, label=label, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", plot)
    y = np.linspace(0.1, 0.2, 5)
    plot_coverage_test(y, y, np.full(5, 0.01), out_path=tmp_path / "c.png", title="t",
                       predicted_label="MFGP mean", show_predicted=show)
    assert ("MFGP mean" in labels) == show


def test_coverage_plot_order_changes_display_only(tmp_path):
    from viz.dispatch import plot_coverage_test

    rng = np.random.default_rng(0)
    mean = rng.uniform(0.1, 0.2, 30)
    obs = mean + rng.normal(0, 0.01, 30)
    results = {
        order: plot_coverage_test(obs, mean, np.full(30, 0.01), out_path=tmp_path / f"{order}.png",
                                  title="t", order=order)
        for order in ("input", "predicted", "observed")
    }
    assert results["input"] == results["predicted"] == results["observed"]
    with pytest.raises(ValueError, match="order must be"):
        plot_coverage_test(obs, mean, np.full(30, 0.01), out_path=tmp_path / "x.png", title="t",
                           order="random")
