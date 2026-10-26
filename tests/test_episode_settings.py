"""Episode building: voxels per step, events per voxel, context size, joint/disjoint targets."""

import numpy as np
import pytest
from pydantic import ValidationError

from core.surrogate_cnp import split_context_target
from core.surrogates import Episode, build_surrogate, load_surrogate
from core.surrogates.fine_tuning import fine_tuning_stages
from core.target_sampling import RealTargetSampler
from data.pseudo_generator import for_scenario
from schemas.data_models import InputMode, StandardBatch
from schemas.surrogates import NeuralTraining


def batch(n_voxels=4, n_events=60):
    rng = np.random.default_rng(np.random.SeedSequence([3, 0]))
    labels = (rng.random((n_voxels, n_events)) < 0.2).astype(int)
    phi = np.broadcast_to(np.arange(n_events)[None, :, None], (n_voxels, n_events, 1))
    return StandardBatch(
        mode=InputMode.FULL,
        theta=np.arange(n_voxels, dtype=float)[:, None],
        phi=phi.astype(float).copy(),
        labels=labels,
    )


def test_default_context_is_random_with_half_cap():
    training = NeuralTraining(n_events=750)
    assert training.context.size == "random"
    assert training.targets == "disjoint"
    assert training.context_range() == (16, 375)
    # The old default 16-64 for 128 events is unchanged.
    assert NeuralTraining().context_range() == (16, 64)


def test_context_cap_options():
    capped = NeuralTraining(n_events=750, context={"size": "random", "max_fraction": 1 / 3})
    assert capped.context_range() == (16, 250)
    uncapped = NeuralTraining(n_events=750, context={"size": "random", "max_fraction": None})
    assert uncapped.context_range() == (16, 748)
    joint = NeuralTraining(n_events=750, targets="joint", context={"max_fraction": None})
    assert joint.context_range() == (16, 749)
    absolute = NeuralTraining(n_events=750, context={"n_min": 8, "n_max": 100})
    assert absolute.context_range() == (8, 100)
    fixed = NeuralTraining(n_events=750, context={"size": "fixed", "n": 250})
    assert fixed.context_range() == (250, 250)


def test_legacy_context_fields_keep_absolute_range():
    training = NeuralTraining(n_events=128, n_context_min=8, n_context_max=40)
    assert training.context_range() == (8, 40)
    # The absolute cap survives a dump/reload and a different fine-tuning n_events.
    reloaded = NeuralTraining.model_validate(training.model_dump())
    assert reloaded.context_range(512) == (8, 40)
    assert NeuralTraining(n_context_max=30).context_range() == (16, 30)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_events": 750, "context": {"size": "fixed", "n": 749}},
        {"n_events": 20, "context": {"n_min": 16}},  # cap 10 < n_min
        {"context": {"max_fraction": 0.5, "n_max": 10}},
        {"context": {"size": "fixed"}},
        {"context": {"size": "fixed", "n": 4, "n_min": 2}},
        {"targets": "joint", "sampling": {"strategy": "class_aware_mixup", "mixup": {}}},
        {
            "targets": "joint",
            "sampling": {"strategy": "positive_quota", "positive_fraction": 0.2},
            "weighting": {"strategy": "sampling_correction"},
        },
        {"targets": "both"},
        {"n_context_min": 4, "context": {"size": "fixed", "n": 4}},
    ],
)
def test_invalid_episode_settings_rejected(kwargs):
    with pytest.raises(ValidationError):
        NeuralTraining(**kwargs)


def test_joint_targets_contain_context_and_same_events_as_disjoint():
    b = batch()
    kwargs = dict(seed=7, batch_size=6, n_events=30, n_context_min=5, n_context_max=12)
    disjoint = RealTargetSampler(b, **kwargs)
    joint = RealTargetSampler(b, joint=True, **kwargs)
    for _ in range(5):
        context_d, target_d, _ = disjoint.next()
        context_j, target_j, weights_j = joint.next()
        np.testing.assert_array_equal(context_d.phi, context_j.phi)
        n_context = context_j.labels.shape[1]
        assert target_j.labels.shape[1] == 30
        assert target_d.labels.shape[1] == 30 - n_context
        np.testing.assert_array_equal(weights_j, np.ones_like(target_j.labels))
        np.testing.assert_array_equal(target_j.labels[:, :n_context], context_j.labels)
        np.testing.assert_array_equal(target_j.labels[:, n_context:], target_d.labels)
        for record in joint.last_provenance:
            assert set(record["context"]) <= set(record["target"])
            assert len(np.unique(record["target"])) == len(record["target"])


def test_joint_context_may_leave_one_new_target():
    sampler = RealTargetSampler(
        batch(), seed=0, batch_size=2, n_events=10, n_context_min=9, n_context_max=9, joint=True
    )
    context, target, _ = sampler.next()
    assert context.labels.shape[1] == 9 and target.labels.shape[1] == 10
    with pytest.raises(ValueError):
        RealTargetSampler(
            batch(), seed=0, batch_size=2, n_events=10, n_context_min=9, n_context_max=9
        )


def test_fixed_context_size_in_every_step():
    training = NeuralTraining(n_events=40, batch_size=3, context={"size": "fixed", "n": 12})
    low, high = training.context_range()
    sampler = RealTargetSampler(
        batch(),
        seed=1,
        batch_size=training.batch_size,
        n_events=training.n_events,
        n_context_min=low,
        n_context_max=high,
    )
    for _ in range(5):
        context, target, _ = sampler.next()
        assert context.labels.shape == (3, 12)
        assert target.labels.shape == (3, 28)


def test_random_context_size_covers_range_once_per_step():
    sampler = RealTargetSampler(
        batch(), seed=2, batch_size=4, n_events=40, n_context_min=3, n_context_max=6
    )
    sizes = set()
    for _ in range(60):
        context, target, _ = sampler.next()
        assert context.labels.shape[0] == 4  # one size shared by all voxels of a step
        assert context.labels.shape[1] + target.labels.shape[1] == 40
        sizes.add(context.labels.shape[1])
    assert sizes == {3, 4, 5, 6}


@pytest.mark.parametrize("targets", ["disjoint", "joint"])
def test_training_records_targets_and_context_range(targets, tmp_path):
    source = for_scenario("S1", seed=0)
    train = source.generate(n_trials=5, n_events=24)
    val = source.generate(n_trials=3, n_events=24, seed=3)
    context, target = split_context_target(val, 4, seed=7)
    settings = NeuralTraining(
        n_steps=4,
        eval_every=2,
        batch_size=3,
        n_events=20,
        targets=targets,
        context={"size": "random", "n_min": 2, "max_fraction": 0.25},
    )
    spec = {"kind": "cnp", "encoder": {"type": "mlp", "latent_dim": 8, "hidden_dims": [12]}}
    model = build_surrogate(spec, source.dim_theta, source.dim_phi)
    report = model.fit(
        train, settings, validation=Episode(context, target), checkpoints=tmp_path / "run"
    )
    audit = report.sampling_audit
    assert audit["targets"] == targets and audit["context_range"] == [2, 5]
    # Joint scores all 20 events per voxel; disjoint only the non-context ones.
    per_step = audit["events"] / (settings.n_steps * settings.batch_size)
    assert per_step == 20 if targets == "joint" else 15 <= per_step <= 18
    restored = load_surrogate(tmp_path / "run/best")
    assert restored.metadata["training"]["targets"] == targets
    assert restored.metadata["training"]["context"]["max_fraction"] == 0.25


def test_n_passes_resolve_to_steps():
    events = 1999 * 1500
    disjoint = NeuralTraining(n_passes=10, n_events=750)  # context 16-375: 554.5 targets/voxel
    assert disjoint.n_steps is None
    assert disjoint.target_events_per_step() == 16 * 554.5
    assert disjoint.resolve_n_steps(events) == 3380
    joint = NeuralTraining(n_passes=10, n_events=750, targets="joint")
    assert joint.resolve_n_steps(events) == 2499  # ceil(10 * 2998500 / 12000)
    fixed = NeuralTraining(n_passes=1, n_events=750, context={"size": "fixed", "n": 250})
    assert fixed.resolve_n_steps(8000) == 1  # 8000 events, 16 * 500 targets per step
    assert NeuralTraining().resolve_n_steps(events) == 10000
    assert NeuralTraining(n_steps=7).resolve_n_steps(events) == 7


def test_n_steps_and_n_passes_are_exclusive():
    with pytest.raises(ValidationError, match="either n_steps or n_passes"):
        NeuralTraining(n_steps=100, n_passes=2)
    with pytest.raises(ValidationError):
        NeuralTraining(n_passes=0)


def test_training_with_n_passes_and_fine_tuning_stage(tmp_path):
    source = for_scenario("S1", seed=0)
    train = source.generate(n_trials=5, n_events=24)  # 120 events
    val = source.generate(n_trials=3, n_events=24, seed=3)
    context, target = split_context_target(val, 4, seed=7)
    settings = NeuralTraining(
        n_passes=2,
        eval_every=2,
        batch_size=3,
        n_events=20,
        context={"size": "fixed", "n": 5},
    )  # 45 targets per step -> ceil(2 * 120 / 45) = 6 steps
    spec = {"kind": "cnp", "encoder": {"type": "mlp", "latent_dim": 8, "hidden_dims": [12]}}
    model = build_surrogate(spec, source.dim_theta, source.dim_phi)
    report = model.fit(
        train, settings, validation=Episode(context, target), checkpoints=tmp_path / "run"
    )
    audit = report.sampling_audit
    assert audit["n_steps"] == 6 and report.history[-1]["step"] == 6
    assert audit["requested_passes"] == 2
    assert audit["target_passes"] == pytest.approx(6 * 45 / 120)
    staged = NeuralTraining.model_validate({**settings.model_dump(), "fine_tuning": {"n_steps": 3}})
    _, fine = fine_tuning_stages(staged, "cnp")
    assert fine.training.n_steps == 3 and fine.training.n_passes is None
