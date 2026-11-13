"""Score stage: predictive scores of a finished last-layer model on prepared sets."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from core.surrogates.datasets import NpzBatches
from data.pseudo_generator import for_scenario
from schemas.surrogates import ScoreRunConfig, SurrogateRunConfig


class NormalizedNpz(NpzBatches):
    """Npz batches with an (identity) normalization, as inference needs; no transfer rule."""

    def normalization(self):
        return {
            g: {"method": "standard", "offset": [0.0], "scale": [1.0]} for g in ("theta", "phi")
        }


class MovingNpz(NormalizedNpz):
    """The same batches with a transfer rule: event inputs relative to the design point."""

    calls = []

    def transfer_events(self, phi, source, source_rows, target):
        MovingNpz.calls.append((source, np.array(source_rows), target))
        return phi


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from core.surrogates.experiment import run_experiment

    tmp = tmp_path_factory.mktemp("score_stage")
    root = tmp / "data"
    for split, fidelity, seed in [
        ("train", "lf", 1),
        ("validation", "lf", 2),
        ("validation", "hf", 3),
        ("test", "lf", 4),
        ("bench", "hf", 5),
    ]:
        batch = for_scenario("S1", seed=seed).generate(n_trials=5, n_events=40, seed=seed)
        folder = root / "batches" / split
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{fidelity}.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )
    config = SurrogateRunConfig(
        model={"kind": "mlp", "architecture": {"hidden_dims": [8]}, "last_layer": {"draws": 200}},
        training=dict(backend="neural", n_steps=4, eval_every=2, batch_size=2, n_events=12),
        data_directory=root,
        data_provider="tests.test_score_stage:NormalizedNpz",
        output_directory=tmp / "run",
        validation_context_events=4,
        selection="bernoulli_log_loss",
    )
    return run_experiment(config), root, tmp


def _config(run, tmp, name, **settings):
    return ScoreRunConfig(run=run, output_directory=tmp / name, **settings)


def test_own_events_by_default_scored_like_the_binomial_gp(run):
    from core.surrogates.inference import open_for_prediction, predict_design_points
    from core.surrogates.readout import predictive_scores
    from core.surrogates.score_stage import run_score

    output, root, tmp = run
    result = run_score(_config(output, tmp, "own", sets=["bench/hf"], threads=2))
    report = json.loads((result / "score.json").read_text())
    (row,) = report["rows"]
    assert row["event_inputs"] == "own" and row["partition"] == "bench"
    assert row["design_points"] == 5 and row["events"] == 200
    # Equal to scoring the plain prediction directly.
    model, provider, _ = open_for_prediction(output / "checkpoints/best", root)
    predicted = predict_design_points(model, provider, "bench", "hf")
    target, _ = provider.load("bench", "hf")
    hits = np.where(target.event_mask, target.labels, 0).sum(axis=1)
    expected, _ = predictive_scores(
        hits,
        target.events_per_design_point,
        predicted["uncertainty"]["probability_draws"].astype(float),
        seed=model.config.last_layer.seed,
        plugin=predicted["prediction"].mean,
    )
    for key in ("predictive_nll", "plugin_count_nll", "mae", "crps"):
        assert row[key] == pytest.approx(expected[key])
    for suffix in ("prediction.npz", "predictive.npz", "coverage.png", "zscore.png"):
        assert (result / f"bench_hf_best_{suffix}").exists()
    assert (result / "compute.json").exists()
    with np.load(result / "bench_hf_best_prediction.npz") as arrays:
        assert (arrays["donor_index"] == -1).all()
        assert arrays["probability_draws"].shape == (200, 5)
    with pytest.raises(FileExistsError):
        run_score(_config(output, tmp, "own", sets=["bench/hf"]))


def test_donors_need_a_transfer_rule_and_never_draw_the_point_itself(run):
    from core.surrogates.score_stage import run_score

    output, _, tmp = run
    with pytest.raises(ValueError, match="transfer_events"):
        run_score(_config(output, tmp, "no_rule", sets=["bench/hf"], donors="scored"))
    MovingNpz.calls.clear()
    result = run_score(
        _config(
            output,
            tmp,
            "donors",
            sets=["bench/hf", "validation/hf"],
            donors="scored",
            data_provider="tests.test_score_stage:MovingNpz",
            seed=3,
        )
    )
    rows = json.loads((result / "score.json").read_text())["rows"]
    for row in rows:
        record = row["event_inputs"]
        assert record["donor_partition"] == row["partition"] and record["own_excluded"]
        assert all(d != i for i, d in enumerate(record["donor_index"]))
    assert [(c[0], c[2]) for c in MovingNpz.calls] == [
        (("bench", "hf"), ("bench", "hf")),
        (("validation", "hf"), ("validation", "hf")),
    ]


def test_training_sets_test_blinding_and_names(run):
    from core.surrogates.score_stage import run_score

    output, _, tmp = run
    with pytest.raises(ValueError, match="trained on"):
        run_score(_config(output, tmp, "train", sets=["train/lf"]))
    with pytest.raises(ValueError, match="unblind_test"):
        run_score(_config(output, tmp, "test", sets=["test/lf"]))
    run_score(_config(output, tmp, "test_open", sets=["test/lf"], unblind_test=True))
    for bad in ({"sets": ["bench"]}, {"sets": ["a/b", "a/b"]}, {"sets": ["a/b"], "donors": "x"}):
        with pytest.raises(ValidationError):
            ScoreRunConfig(run=output, output_directory=tmp / "x", **bad)


def test_cli_and_config_paths(run, tmp_path):
    import subprocess
    import sys

    import yaml

    from schemas.surrogates import load_score_config

    output, _, _ = run
    path = tmp_path / "score.yaml"
    path.write_text(
        yaml.safe_dump({"run": str(output), "sets": ["bench/hf"], "output_directory": "scored"})
    )
    config = load_score_config(path)
    assert config.output_directory == tmp_path / "scored" and config.donors is None
    done = subprocess.run(
        [sys.executable, "-m", "core.surrogates", "score", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "scored" / "score.json").exists()
