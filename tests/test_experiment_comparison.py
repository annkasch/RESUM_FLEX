import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.experiments.comparison import compare, compare_saved, merge, seeded
from core.experiments.runner import run
from core.experiments.storage import RunStore
from tests.test_experiment_runner import config


def test_preflight_rejects_different_evaluation_before_running(tmp_path):
    cfg = config(tmp_path)
    baseline = tmp_path / "base.yaml"
    baseline.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    settings = dict(
        name="bad",
        baseline=str(baseline),
        seeds=[0],
        variants=[dict(name="base"), dict(name="different", overrides={"evaluation": {"seed": 3}})],
    )
    file = tmp_path / "compare.yaml"
    file.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="share dataset"):
        compare(file)
    assert not (cfg.store / "runs").exists()


def test_seeded_stages_preserve_offsets_and_model_discriminator(tmp_path):
    cfg = config(tmp_path)
    training = cfg.pipeline.training
    from schemas.surrogates import TrainingStage

    cfg.pipeline.stages = [
        TrainingStage(name="base", training=training),
        TrainingStage(
            name="fine", start_from="base/best", training=training.model_copy(update={"seed": 1})
        ),
    ]
    result = seeded(cfg, 7)
    assert [s.training.seed for s in result.pipeline.stages] == [7, 8]
    assert cfg.pipeline.training.seed == 0
    assert merge({"pipeline": {"kind": "event", "model": "x"}}, {"pipeline": {"kind": "count_gp"}})[
        "pipeline"
    ] == {"kind": "count_gp"}


def test_checkpoint_mfgp_comparison_without_network_retraining(tmp_path, monkeypatch):
    pytest.importorskip("GPy")
    cfg = config(tmp_path)
    from schemas.surrogates import LastLayerFineTuning

    cfg.pipeline.training.fine_tuning = LastLayerFineTuning(n_steps=2)
    from schemas.experiments import ExperimentSpec

    cfg = ExperimentSpec.model_validate(cfg.model_dump())
    trained = run(cfg)
    base = cfg.model_copy(deep=True)
    base.pipeline.fit = False
    base.pipeline.initial_checkpoint = trained / "backend/checkpoints/pretraining"
    from schemas.surrogates import MFGPStageConfig

    base.pipeline.spatial_regression = MFGPStageConfig(n_restarts=1, n_context=4)
    base.evaluation.plots = []
    baseline = tmp_path / "base.yaml"
    baseline.write_text(yaml.safe_dump(base.model_dump(mode="json")))
    file = tmp_path / "comparison.yaml"
    file.write_text(
        yaml.safe_dump(
            dict(
                name="before-after",
                baseline=str(baseline),
                seeds=[0],
                variants=[
                    dict(name="before"),
                    dict(
                        name="after",
                        overrides={
                            "pipeline": {
                                "initial_checkpoint": str(trained / "backend/checkpoints/best")
                            }
                        },
                    ),
                ],
            )
        )
    )
    from core.surrogates import training

    def no_training(*args, **kwargs):
        raise AssertionError("Downstream comparison must not retrain the CNP")

    monkeypatch.setattr(training, "_fit_stage", no_training)
    out = compare(file)
    store = RunStore(cfg.store)
    assert store.inspect(out.name)["status"] == "completed"
    children = json.loads((out / "children.json").read_text())
    assert len(children) == 2
    arrays = [np.load(Path(c["path"]) / "backend/mfgp/training_arrays.npz") for c in children]
    for key in ("X_lf", "X_hf", "Y_hf_raw"):
        np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
    assert not np.array_equal(arrays[0]["Y_lf_cnp"], arrays[1]["Y_lf_cnp"])
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["paired_differences"]) == 2
    for c in children:
        meta = json.loads((Path(c["path"]) / "backend/mfgp/model.json").read_text())
        assert str(trained) in meta["surrogate_checkpoint"]
    again = compare_saved(cfg.store, [Path(c["path"]).name for c in children])
    assert (again / "summary.json").exists()


def test_incompatible_saved_targets_are_separate_groups(tmp_path):
    from core.experiments.comparison import summarize

    children = []
    for i, target in enumerate(["target_a", "target_b"]):
        folder = tmp_path / str(i)
        folder.mkdir()
        (folder / "metrics.json").write_text(
            json.dumps(
                [
                    dict(
                        name="count_gp_validation_hf",
                        model="count_gp",
                        partition="validation/hf",
                        target_id=target,
                        metrics={"mae": 0.1 + i},
                        checkpoint="saved",
                    )
                ]
            )
        )
        children.append(dict(variant=str(i), seed=0, path=str(folder)))
    summarize(children, tmp_path)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert len(result["groups"]) == 2
    assert result["paired_differences"] == []
