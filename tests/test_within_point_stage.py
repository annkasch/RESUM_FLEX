"""Within-design-point metrics on a finished run: equal the read-out of the same run."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from data.pseudo_generator import for_scenario
from schemas.surrogates import SurrogateRunConfig, WithinPointRunConfig

WITHIN_KEYS = (
    "within_point_log_loss_gain",
    "within_point_log_loss_skill",
    "within_point_auc",
    "within_point_spread_z",
    "within_point_log_loss_gain_ci68",
    "within_point_log_loss_gain_ci95",
    "within_point_auc_ci68",
    "within_point_auc_ci95",
    "within_point_gain_coverage",
)


@pytest.fixture(scope="module")
def training_run(tmp_path_factory):
    from core.surrogates.experiment import run_experiment

    tmp = tmp_path_factory.mktemp("within_point_stage")
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
        within_point_null_draws=200,
    )
    return run_experiment(config), tmp


def _readout(rows):
    return {
        (r["checkpoint"], r["split"], r["fidelity"]): {k: r[k] for k in WITHIN_KEYS}
        for r in rows
        if r["split"] != "train"
    }


def _added(run):
    return [run / "within_point.json", *run.glob("*_*_*_within_point*")]


def _remove_added(run):
    for path in _added(run):
        path.unlink(missing_ok=True)


def test_stage_reproduces_the_training_readout(training_run):
    from core.surrogates.within_point_stage import run_within_point

    run, _ = training_run
    _remove_added(run)
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    config = WithinPointRunConfig(
        run=run, checkpoints=["best", "final"], within_point_null_draws=200
    )
    assert run_within_point(config) == run  # an added evaluation, written into the run
    report = json.loads((run / "within_point.json").read_text())
    expected = _readout(json.loads((run / "metrics.json").read_text()))
    assert set(_readout(report["rows"])) == set(expected)  # validation + unblinded test
    assert _readout(report["rows"]) == expected
    assert all(v["within_point_auc"] is not None for v in expected.values())
    assert all(r["step"] is not None and r["positives"] > 0 for r in report["rows"])
    with (
        np.load(run / "lf_validation_best_within_point.npz") as ours,
        np.load(run / "lf_validation_best.npz") as theirs,
    ):
        for key in ours.files:
            assert np.array_equal(ours[key], theirs[key]), key
    assert (run / "lf_test_final_within_point_gain.png").exists()
    assert all(path.read_bytes() == data for path, data in before.items())  # run files kept
    with pytest.raises(FileExistsError):
        run_within_point(config)


def test_stage_rejects_changed_data_and_unknown_sets(training_run):
    from core.surrogates.within_point_stage import run_within_point

    run, tmp = training_run
    _remove_added(run)
    with pytest.raises(ValueError, match="did not evaluate"):
        run_within_point(WithinPointRunConfig(run=run, sets=["validation/hf"]))
    path = tmp / "data" / "batches" / "validation" / "lf.npz"
    original = path.read_bytes()
    with np.load(path) as arrays:
        changed = dict(arrays)
    changed["labels"] = 1 - changed["labels"]
    np.savez(path, **changed)
    try:
        with pytest.raises(ValueError, match="data changed"):
            run_within_point(WithinPointRunConfig(run=run))
    finally:
        path.write_bytes(original)
    assert not any(p.exists() for p in _added(run))  # refused before writing


def test_config_checks_set_names():
    with pytest.raises(ValidationError):
        WithinPointRunConfig(run="r", sets=["validation/stream"])
    with pytest.raises(ValidationError):
        WithinPointRunConfig(run="r", sets=["train/lf"])
    with pytest.raises(ValidationError):
        WithinPointRunConfig(run="r", within_point_null_draws=10)
    with pytest.raises(ValidationError):  # writes into the run, no output directory
        WithinPointRunConfig(run="r", output_directory="o")
