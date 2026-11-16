"""Learning-rate schedule of neural training and the --run-seed override (synthetic data)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.surrogate_cnp import split_context_target
from core.surrogates import Episode, build_surrogate
from data.pseudo_generator import for_scenario
from schemas.surrogates import (
    CosineSchedule,
    NeuralTraining,
    ScoreRunConfig,
    SurrogateRunConfig,
    score_with_run_seed,
    with_run_seed,
)


def test_cosine_factor_runs_from_one_to_final_fraction():
    schedule = CosineSchedule(final_fraction=0.01)
    factors = [schedule.factor(step, 11) for step in range(1, 12)]
    assert factors[0] == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(0.01)
    assert factors[5] == pytest.approx(0.505)  # midpoint: halfway between 1 and 0.01
    assert all(a > b for a, b in zip(factors, factors[1:], strict=False))
    assert schedule.factor(1, 1) == 1.0


def test_schedule_defaults_to_constant_and_rejects_bad_fraction():
    assert NeuralTraining().lr_schedule.kind == "constant"
    with pytest.raises(ValidationError):
        NeuralTraining(lr_schedule={"kind": "cosine", "final_fraction": 1.5})
    with pytest.raises(ValidationError):
        NeuralTraining(lr_schedule={"kind": "linear"})


def _fit(schedule):
    source = for_scenario("S1", seed=7)
    batch = source.generate(n_trials=4, n_events=24)
    batch.labels[:, :4] = 1
    context, target = split_context_target(batch, 4, seed=3)
    model = build_surrogate(
        {"kind": "mlp", "architecture": {"hidden_dims": [12]}}, source.dim_theta, source.dim_phi
    )
    training = NeuralTraining(
        n_steps=5, eval_every=1, batch_size=3, n_events=12, learning_rate=0.01,
        lr_schedule=schedule,
    )  # fmt: skip
    report = model.fit(batch, training, validation=Episode(context, target))
    return [row["learning_rate"] for row in report.history]


def test_history_records_the_learning_rate_of_each_step():
    assert _fit({"kind": "constant"}) == pytest.approx([0.01] * 5)
    rates = _fit({"kind": "cosine", "final_fraction": 0.1})
    expected = [0.01 * CosineSchedule(final_fraction=0.1).factor(s, 5) for s in range(1, 6)]
    assert rates == pytest.approx(expected)
    assert rates[0] == pytest.approx(0.01) and rates[-1] == pytest.approx(0.001)


def _run_config(output):
    return SurrogateRunConfig(
        model={"kind": "mlp", "architecture": {"hidden_dims": [12]}, "last_layer": {}},
        training=NeuralTraining(seed=0),
        data_directory=Path("/data"),
        output_directory=output,
    )


def test_run_seed_sets_model_seeds_and_seed_directory():
    config = _run_config(Path("/runs/model/v1/seed0"))
    seeded = with_run_seed(config, 2)
    assert seeded.output_directory == Path("/runs/model/v1/seed2")
    assert seeded.training.seed == 2
    assert seeded.model.last_layer.seed == 2
    assert seeded.validation_seed == config.validation_seed  # same selection subset
    assert config.training.seed == 0  # the loaded config is unchanged


@pytest.mark.parametrize("output", ["/runs/model/v1", "/runs/seed0/v1/seed0"])
def test_run_seed_needs_exactly_one_seed_directory(output):
    with pytest.raises(ValueError, match="seed<k>"):
        with_run_seed(_run_config(Path(output)), 1)


def test_score_run_seed_moves_run_and_output_keeps_donor_seed():
    config = ScoreRunConfig(
        run=Path("/runs/model/v1/seed0"),
        sets=["validation/hf"],
        output_directory=Path("/runs/model/v1/seed0/score"),
        seed=5,
    )
    seeded = score_with_run_seed(config, 1)
    assert seeded.run == Path("/runs/model/v1/seed1")
    assert seeded.output_directory == Path("/runs/model/v1/seed1/score")
    assert seeded.seed == 5
