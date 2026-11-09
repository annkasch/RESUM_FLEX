"""Checkpoint selection on HF validation and by the voxel count NLL."""

import json

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import binom

from core.surrogates import build_surrogate, evaluate_surrogate
from data.pseudo_generator import for_scenario
from schemas.surrogates import SurrogateRunConfig


def _write(root, sets):
    for (split, fidelity), seed in sets.items():
        batch = for_scenario("S1", seed=seed).generate(n_trials=4, n_events=30, seed=seed)
        folder = root / "batches" / split
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{fidelity}.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )


def test_voxel_count_nll_is_the_binomial_plugin_score():
    batch = for_scenario("S1", seed=0).generate(n_trials=5, n_events=40, seed=0)
    model = build_surrogate({"kind": "mlp", "architecture": {"hidden_dims": [8]}}, 1, 1)
    metrics, arrays = evaluate_surrogate(model, batch)
    k, n = batch.labels.sum(axis=1), np.full(5, 40)
    expected = -binom.logpmf(k, n, arrays["predicted"]).mean()
    assert metrics["voxel_count_nll"] == pytest.approx(expected)


def test_hf_selection_uses_hf_validation(tmp_path):
    from core.surrogates.experiment import run_experiment

    _write(tmp_path / "data", {("train", "lf"): 1, ("validation", "hf"): 3})
    config = SurrogateRunConfig(
        model={"kind": "mlp", "architecture": {"hidden_dims": [8]}},
        training=dict(
            backend="neural",
            n_steps=6,
            eval_every=2,
            batch_size=2,
            n_events=12,
        ),
        data_directory=tmp_path / "data",
        output_directory=tmp_path / "run",
        lf_validation=False,
        selection_fidelity="hf",
        selection="voxel_count_nll",
        validation_context_events=4,
    )
    output = run_experiment(config)
    experiment = json.loads((output / "experiment.json").read_text())
    assert experiment["selection_split"] == "validation/hf"
    history = json.loads((output / "history.json").read_text())
    best = json.loads((output / "checkpoints/best/model.json").read_text())["metadata"]["step"]
    scores = {r["step"]: r["voxel_count_nll"] for r in history}
    assert best == min(scores, key=scores.get)
    rows = json.loads((output / "metrics.json").read_text())
    assert {(r["split"], r["fidelity"]) for r in rows} == {("train", "lf"), ("validation", "hf")}


def test_hf_selection_needs_hf_validation(tmp_path):
    with pytest.raises(ValidationError, match="hf_validation"):
        SurrogateRunConfig(
            model={"kind": "mlp"},
            training={"backend": "neural"},
            data_directory=tmp_path,
            output_directory=tmp_path / "run",
            hf_validation=False,
            selection_fidelity="hf",
        )
