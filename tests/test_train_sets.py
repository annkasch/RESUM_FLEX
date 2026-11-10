"""Event-model training on several sets (train_sets): concatenation, schema and pipeline."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from schemas.data_models import InputMode, StandardBatch, concatenate_batches
from schemas.surrogates import SurrogateRunConfig


def _batch(rng, b, n, counts=None):
    labels = rng.binomial(1, 0.3, (b, n)).astype(np.int8)
    if counts is not None:
        labels[np.arange(n) >= np.asarray(counts)[:, None]] = 0
    return StandardBatch(
        mode=InputMode.FULL,
        theta=rng.normal(size=(b, 2)),
        phi=rng.normal(size=(b, n, 2)).astype(np.float32),
        labels=labels,
        event_counts=None if counts is None else np.asarray(counts),
    )


def test_concatenate_pads_and_counts():
    rng = np.random.default_rng(0)
    a, b = _batch(rng, 2, 4), _batch(rng, 3, 6, counts=[6, 5, 2])
    out = concatenate_batches([a, b])
    assert out.labels.shape == (5, 6) and out.phi.shape == (5, 6, 2)
    np.testing.assert_array_equal(out.events_per_design_point, [4, 4, 6, 5, 2])
    np.testing.assert_array_equal(out.theta, np.concatenate([a.theta, b.theta]))
    np.testing.assert_array_equal(out.labels[:2, :4], a.labels)
    assert not out.labels[:2, 4:].any() and not out.phi[:2, 4:].any()
    # Event means are unchanged by the padding.
    values = out.labels.astype(float)
    np.testing.assert_allclose(out.design_point_mean(values)[:2], a.labels.mean(axis=1))
    # Equal full event axes stay rectangular.
    assert concatenate_batches([a, a]).event_counts is None


def test_concatenate_rejects_mismatches():
    rng = np.random.default_rng(1)
    a = _batch(rng, 2, 4)
    other = StandardBatch(mode=InputMode.DESIGN_ONLY, theta=a.theta, labels=a.labels)
    with pytest.raises(ValueError, match="modes"):
        concatenate_batches([a, other])
    wider = StandardBatch(
        mode=InputMode.FULL, theta=a.theta, phi=np.zeros((2, 4, 3)), labels=a.labels
    )
    with pytest.raises(ValueError, match="phi dimensions"):
        concatenate_batches([a, wider])
    with pytest.raises(ValueError, match="at least one"):
        concatenate_batches([])


BASE = dict(
    model={"kind": "mlp", "architecture": {"hidden_dims": [8]}},
    training=dict(
        backend="neural",
        n_steps=4,
        eval_every=2,
        batch_size=2,
        n_events=6,
    ),
    data_directory="d",
    output_directory="o",
    partitions={"train": "train", "validation": "validation", "test": "test"},
)


def test_train_sets_schema():
    default = SurrogateRunConfig(**BASE)
    assert default.training_sets == [("train", "lf")] and default.training_label == "lf"
    mixed = SurrogateRunConfig(
        **BASE,
        train_sets=[{"partition": "lf2", "fidelity": "lf"}, {"partition": "hft", "fidelity": "hf"}],
    )
    assert mixed.training_sets == [("lf2", "lf"), ("hft", "hf")]
    assert mixed.training_label == "mixed"
    for sets, message in [
        ([{"partition": "validation", "fidelity": "hf"}], "validation or test"),
        ([{"partition": "test", "fidelity": "lf"}], "validation or test"),
        ([{"partition": "a", "fidelity": "lf"}] * 2, "unique"),
    ]:
        with pytest.raises(ValidationError, match=message):
            SurrogateRunConfig(**BASE, train_sets=sets)
    with pytest.raises(ValidationError):
        SurrogateRunConfig(**BASE, train_sets=[])


def test_mixed_train_sets_run_through_the_pipeline(tmp_path):
    from core.surrogates.experiment import run_experiment

    rng = np.random.default_rng(2)
    root = tmp_path / "data"
    shapes = {("lf2", "lf"): (4, 10), ("hft", "hf"): (3, 25), ("validation", "lf"): (3, 10)}
    for (partition, fidelity), (b, n) in shapes.items():
        batch = _batch(rng, b, n)
        folder = root / "batches" / partition
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{fidelity}.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )
    config = SurrogateRunConfig(
        **{
            **BASE,
            "model": {**BASE["model"], "last_layer": {"draws": 100}},
            "data_directory": root,
            "output_directory": tmp_path / "run",
        },
        train_sets=[{"partition": "lf2", "fidelity": "lf"}, {"partition": "hft", "fidelity": "hf"}],
        hf_validation=False,
        validation_context_events=2,
        selection="bernoulli_log_loss",
    )
    output = run_experiment(config)
    experiment = json.loads((output / "experiment.json").read_text())
    assert set(experiment["data"]["train/mixed"]) == {"lf2/lf", "hft/hf"}
    rows = json.loads((output / "metrics.json").read_text())
    train = [r for r in rows if r["split"] == "train"]
    assert train and all(r["fidelity"] == "mixed" for r in train)
    assert {r["fidelity"] for r in rows if r["split"] == "validation"} == {"lf"}
    # The last layer saw every valid training event: 4 * 10 LF + 3 * 25 HF.
    manifest = json.loads((output / "checkpoints/best/model.json").read_text())
    assert manifest["metadata"]["last_layer"]["events"] == 4 * 10 + 3 * 25
