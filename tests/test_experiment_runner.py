import json

import numpy as np
import pytest

from core.experiments.reporting import regenerate
from core.experiments.runner import run
from core.experiments.storage import RunStore
from data.pseudo_generator import for_scenario
from schemas.experiments import ExperimentSpec


def config(tmp_path, kind="event"):
    root = tmp_path / "data"
    source = for_scenario("S1", seed=42)
    for split in ("train", "validation"):
        for fid in ("lf", "hf"):
            folder = root / "batches" / split
            folder.mkdir(parents=True, exist_ok=True)
            b = source.generate(n_trials=4, n_events=24)
            np.savez(
                folder / f"{fid}.npz",
                theta=b.theta,
                phi=b.phi,
                labels=b.labels,
                mode=b.mode.value,
                event_ids=np.tile(np.arange(24), (4, 1)),
            )
    return ExperimentSpec(
        name="tiny",
        store=tmp_path / "store",
        dataset={"prepared": root},
        pipeline=(
            {
                "kind": "event",
                "model": {"kind": "cnp", "encoder": {"latent_dim": 8, "hidden_dims": [12]}},
                "training": {
                    "backend": "neural",
                    "n_steps": 2,
                    "eval_every": 1,
                    "batch_size": 2,
                    "n_events": 12,
                    "n_context_min": 4,
                    "n_context_max": 4,
                },
            }
            if kind == "event"
            else {
                "kind": "count_gp",
                "backend": {"n_restarts": 1, "max_iters": 5},
                "prediction_draws": 2048,
            }
        ),
        evaluation={
            "context_events": 4,
            "partitions": ["validation/lf", "validation/hf"],
            "plots": ["means", "coverage"],
        },
    )


def test_event_lifecycle_and_reproducible_report(tmp_path):
    cfg = config(tmp_path)
    out = run(cfg)
    store = RunStore(cfg.store)
    assert store.inspect(out.name)["status"] == "completed"
    before = json.loads((out / "metrics.json").read_text())
    report = regenerate(cfg.store, out.name)
    assert json.loads((report / "metrics.json").read_text()) == before
    store.inspect(out.name)
    assert before[0]["metrics"]["mae"] >= 0
    assert (out / "provenance.json").exists()


def test_count_gp_and_event_share_population_targets(tmp_path):
    pytest.importorskip("GPy")
    cfg = config(tmp_path, "count_gp")
    count = run(cfg)
    # Reuse exact prepared arrays rather than regenerate pseudo data.
    event = cfg.model_copy(deep=True)
    event.pipeline = config(tmp_path / "other").pipeline
    event.store = cfg.store
    neural = run(event)
    a = json.loads((count / "evaluation/manifest.json").read_text())
    b = json.loads((neural / "evaluation/manifest.json").read_text())
    assert a == b
    assert len(RunStore(cfg.store).list(status="completed")) == 2
    assert all(
        r["distribution"]["measure"] == "count_mass"
        for r in json.loads((count / "metrics.json").read_text())
    )


def test_failure_is_recorded(tmp_path):
    cfg = config(tmp_path)
    cfg.evaluation.context_events = 100
    with pytest.raises(ValueError, match="no target"):
        run(cfg)
    failed = RunStore(cfg.store).list(status="failed")
    assert len(failed) == 1
    assert "no target" in failed[0]["error"]
