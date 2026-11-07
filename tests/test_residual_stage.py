"""Residual test on a finished run: equals the read-out of a run trained with the test."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from data.pseudo_generator import for_scenario
from schemas.surrogates import ResidualRunConfig, ResidualTestConfig, SurrogateRunConfig

pytest.importorskip("lightgbm")

RESIDUAL = {"min_positives": 1, "min_data_in_leaf": 5, "max_trees": 30}


def _results(rows):
    return {
        (r["checkpoint"], r["split"], r["fidelity"]): r["residual_test"]
        for r in rows
        if "residual_test" in r
    }


@pytest.fixture(scope="module")
def training_run(tmp_path_factory):
    from core.surrogates.experiment import run_experiment

    tmp = tmp_path_factory.mktemp("residual_stage")
    source = for_scenario("S1", seed=1)
    root = tmp / "data"
    for split, seed in (("train", 1), ("validation", 2), ("test", 3)):
        batch = source.generate(n_trials=6, n_events=80, seed=seed)
        folder = root / "batches" / split
        folder.mkdir(parents=True)
        np.savez(
            folder / "lf.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )
    config = SurrogateRunConfig(
        model={"kind": "mlp", "architecture": {"hidden_dims": [8]}},
        training={
            "backend": "neural",
            "n_steps": 4,
            "eval_every": 2,
            "batch_size": 2,
            "n_events": 40,
        },
        selection="bernoulli_log_loss",
        hf_validation=False,
        test_fidelities=["lf"],
        unblind_test=True,
        data_directory=root,
        output_directory=tmp / "run",
        validation_context_events=4,
        residual_test=RESIDUAL,
    )
    return run_experiment(config), tmp


def test_stage_reproduces_the_training_readout(training_run):
    from core.surrogates.residual_stage import run_residual

    run, tmp = training_run
    config = ResidualRunConfig(
        run=run, checkpoints=["best", "final"], residual_test=RESIDUAL, output_directory=tmp / "r"
    )
    output = run_residual(config)
    report = json.loads((output / "residual_test.json").read_text())
    expected = _results(json.loads((run / "metrics.json").read_text()))
    assert set(_results(report["rows"])) == set(expected)  # validation + unblinded test
    assert _results(report["rows"]) == expected
    assert all(v["gain"] is not None for v in expected.values())
    assert (output / "lf_test_final_residual.npz").exists()
    with pytest.raises(FileExistsError):
        run_residual(config)


def test_stage_rejects_changed_data_and_unknown_sets(training_run):
    from core.surrogates.residual_stage import run_residual

    run, tmp = training_run
    with pytest.raises(ValueError, match="did not evaluate"):
        run_residual(ResidualRunConfig(run=run, sets=["validation/hf"], output_directory=tmp / "a"))
    path = tmp / "data" / "batches" / "validation" / "lf.npz"
    original = path.read_bytes()
    with np.load(path) as arrays:
        changed = dict(arrays)
    changed["labels"] = 1 - changed["labels"]
    np.savez(path, **changed)
    try:
        with pytest.raises(ValueError, match="data changed"):
            run_residual(ResidualRunConfig(run=run, output_directory=tmp / "b"))
    finally:
        path.write_bytes(original)


def test_stage_on_a_stream_run(tmp_path):
    from core.surrogates.residual_stage import run_residual
    from core.surrogates.stream_training import run_stream
    from tests.test_stream_field import _run_config

    config = _run_config(tmp_path, n_passes=1)
    config = config.model_copy(update={"residual_test": ResidualTestConfig(**RESIDUAL)})
    run_stream(config)
    run = tmp_path / "run"
    output = run_residual(
        ResidualRunConfig(run=run, residual_test=RESIDUAL, output_directory=tmp_path / "r")
    )
    rows = json.loads((output / "residual_test.json").read_text())["rows"]
    expected = {
        k: v
        for k, v in _results(json.loads((run / "metrics.json").read_text())).items()
        if k[0] == "best"
    }
    assert _results(rows) == expected
    assert all(v["gain"] is not None for v in expected.values())
    assert {k[2] for k in expected} == {"stream", "lf", "hf"}
    assert expected["best", "validation", "stream"]["units"] == "events"


def test_config_checks_set_names():
    with pytest.raises(ValidationError):
        ResidualRunConfig(run="r", sets=["train/lf"], output_directory="o")
    with pytest.raises(ValidationError):
        ResidualRunConfig(run="r", checkpoints=["best", "best"], output_directory="o")
