"""Stream training of the transformer and the BDT: schema, record mapping, sample, runs."""

import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from core.event_stream import event_labels
from core.surrogates import load_surrogate
from core.surrogates.models import build_surrogate
from core.surrogates.stream_training import (
    case_control_sample,
    check_record_width,
    run_stream,
    stream_logits,
)
from schemas.data_models import InputMode, StandardBatch
from schemas.surrogates import FieldModel, StreamRunConfig, StreamTreeTraining
from tests.test_bernoulli_field import NORMALIZATION, _physical_batch, _random_physical
from tests.test_stream_field import FakeEventSource, _records, _run_config

TRANSFORMER = {
    "kind": "transformer",
    "architecture": {"token_dim": 8, "n_heads": 2, "n_layers": 1, "feedforward_dim": 16},
}
BASE = {"event_store": "s", "event_source": "m:C", "output_directory": "o"}


def _config(model, *, dim_theta=3, dim_phi=6, **extra):
    return StreamRunConfig.model_validate(
        {"model": model, "dim_theta": dim_theta, "dim_phi": dim_phi, **BASE, **extra}
    )


def _variant(tmp_path, model, training):
    """The field's test run (data, evaluation sets) with another model and training."""
    data = _run_config(tmp_path).model_dump(mode="json")
    data.update(model=model, training=training)
    return StreamRunConfig.model_validate(data)


# --- Schema --------------------------------------------------------------------------------


def test_stream_transformer_is_the_unencoded_field_with_a_transformer():
    network = _config(TRANSFORMER).network
    assert network.kind == "field" and network.encoding.kind == "none"
    assert network.architecture.token_dim == 8 and network.last_layer is None  # not configured
    assert network.encoding.columns == []


def test_training_settings_follow_the_kind():
    trees = _config({"kind": "bdt"}, training={"negatives_per_positive": 5})
    assert isinstance(trees.training, StreamTreeTraining)
    assert trees.training.negatives_per_positive == 5
    with pytest.raises(ValidationError, match="training of a stream bdt"):
        _config({"kind": "bdt"}, training={"n_passes": 3})  # gradient steps: not for trees
    with pytest.raises(ValidationError, match="training of a stream mlp"):
        _config({"kind": "mlp"}, training={"negatives_per_positive": 5})
    with pytest.raises(ValidationError):
        _config({"kind": "cnp"})  # no stream CNP: ungrouped events have no context sets


def test_record_width_is_dim_phi_plus_the_label():
    check_record_width(np.zeros((4, 7), dtype=np.float32), 6, "Training")
    with pytest.raises(ValueError, match=r"expected \[M, dim_phi \+ 1\] = \[M, 7\]"):
        check_record_width(np.zeros((4, 8), dtype=np.float32), 6, "Training")


def test_a_field_architecture_is_an_mlp_unless_it_names_transformer_settings():
    assert hasattr(FieldModel().architecture, "hidden_dims")
    assert hasattr(FieldModel(architecture={}).architecture, "hidden_dims")
    assert FieldModel(architecture={"token_dim": 8, "n_heads": 2}).architecture.token_dim == 8


# --- Transformer ---------------------------------------------------------------------------


def test_stream_transformer_logits_match_on_design_point_inputs():
    dim_theta, dim_phi = 2, 3
    config = _config(TRANSFORMER, dim_theta=dim_theta, dim_phi=dim_phi)
    rng = np.random.default_rng(4)
    b, n = 3, 20
    phi = rng.normal(-1.0, 0.5, (b, n, dim_phi))  # physical units
    records = phi.reshape(b * n, dim_phi)
    data_norm = {
        "theta": {"offset": [4.0] * dim_theta, "scale": [3.0] * dim_theta},
        "phi": {"offset": [-0.5] * dim_phi, "scale": [0.7] * dim_phi},
    }
    surrogate = build_surrogate(config.network, dim_theta, dim_phi, seed=0)
    surrogate.module.set_stream_normalization(
        {"offset": list(records.mean(0)), "scale": list(records.std(0))}
    )
    surrogate.configure_normalization(data_norm)
    batch = StandardBatch(
        mode=InputMode.FULL,
        theta=rng.normal(size=(b, dim_theta)),  # not an input of a stream model
        phi=((phi - (-0.5)) / 0.7).astype(np.float32),
        labels=np.zeros((b, n), dtype=np.int8),
    )
    stream = torch.as_tensor(
        np.concatenate([records, np.zeros((b * n, 1))], axis=1), dtype=torch.float32
    )
    surrogate.module.eval()
    with torch.no_grad():
        design = surrogate.module(None, batch)  # [B, N]
        direct = surrogate.module.forward_events(stream).reshape(b, n)
    torch.testing.assert_close(design, direct, rtol=1e-4, atol=1e-4)
    assert surrogate.output_layer() is surrogate.module.decoder.transformer.head[-1]


def test_stream_transformer_trains_and_scores_on_design_points(tmp_path):
    training = {"batch_size": 256, "n_passes": 1, "eval_every": 10, "learning_rate": 3e-3}
    model = {
        **TRANSFORMER,
        # Chunks of 50 events: the last-layer pass joins several head calls per batch.
        "architecture": {**TRANSFORMER["architecture"], "inference_chunk_size": 50},
        "last_layer": {"prior_precision": "evidence", "draws": 100},
    }
    config = _variant(tmp_path, model, {**training, "device": "cpu"})
    report = run_stream(config)
    assert report["history"][-1]["stream_log_loss"] < report["history"][0]["stream_log_loss"]
    run = tmp_path / "run"
    best = load_surrogate(run / "checkpoints" / "best")
    assert best.config.kind == "field" and best.module.uses_stream_units
    assert hasattr(best.module.decoder, "transformer")
    assert best.last_layer is not None and best.metadata["last_layer"]["events"] == 6000
    experiment = json.loads((run / "experiment.json").read_text())
    assert experiment["resolved_config"]["model"]["kind"] == "transformer"
    rows = json.loads((run / "metrics.json").read_text())
    assert {r["fidelity"] for r in rows} == {"stream", "lf", "hf"}
    hf = next(r for r in rows if r["fidelity"] == "hf" and r["checkpoint"] == "best")
    assert hf["design_point_predictive"] is not None


# --- BDT -----------------------------------------------------------------------------------


class _Labels:
    """Event source of labels only (one input column), for the case-control sample."""

    def __init__(self, labels):
        self.records = _records(np.zeros((len(labels), 3)), np.zeros((len(labels), 3)), labels)
        self.n_events, self.n_positive = len(labels), int(labels.sum())

    def batches(self, batch_size, pass_index=0, **_):
        for start in range(0, self.n_events, batch_size):
            yield torch.as_tensor(self.records[start : start + batch_size])


def test_case_control_sample_keeps_positives_and_weights_negatives():
    rng = np.random.default_rng(0)
    labels = rng.random(200_000) < 0.002
    source = _Labels(labels)
    records, weights, summary = case_control_sample(
        source, negatives_per_positive=10, batch_size=4096, seed=3
    )
    kept = event_labels(torch.as_tensor(records)).numpy() == 1
    assert kept.sum() == labels.sum() == summary["positives"]
    assert np.all(weights[kept] == 1)
    fraction = 10 * labels.sum() / (~labels).sum()
    assert summary["negative_fraction"] == pytest.approx(fraction)
    # Weighted negatives estimate the number of all negatives (binomial noise only).
    assert weights[~kept].sum() == pytest.approx((~labels).sum(), rel=0.03)
    again = case_control_sample(source, negatives_per_positive=10, batch_size=4096, seed=3)
    np.testing.assert_array_equal(again[0], records)
    everything = case_control_sample(source, negatives_per_positive=1e6, batch_size=4096, seed=3)
    assert everything[2]["negative_fraction"] == 1 and len(everything[0]) == len(labels)


def test_stream_records_are_the_physical_phi_columns():
    from core.surrogate_bdt import stream_records

    rng = np.random.default_rng(1)
    centre, point, direction = _random_physical(rng, b=2, n=5)
    batch = _physical_batch(centre, point, direction)
    records = stream_records(batch, NORMALIZATION)
    expected = np.concatenate([point.reshape(-1, 3), direction.reshape(-1, 3)], axis=1)
    np.testing.assert_allclose(records, expected, rtol=1e-5, atol=1e-9)


def test_stream_bdt_trains_selects_and_scores_on_design_points(tmp_path):
    pytest.importorskip("sklearn")
    from core.surrogates.inference import open_for_prediction

    model = {"kind": "bdt", "architecture": {"max_iter": 60, "eval_every": 20, "max_depth": 2}}
    training = {"negatives_per_positive": 4, "batch_size": 1000, "validation_events": 1500}
    config = _variant(tmp_path, model, training)
    report = run_stream(config)
    assert [r["step"] for r in report["history"]] == [20, 40, 60]
    run = tmp_path / "run"
    experiment = json.loads((run / "experiment.json").read_text())
    sample = experiment["training_sample"]
    train = FakeEventSource(tmp_path / "store", "train")
    assert sample["positives"] == train.n_positive and sample["source_events"] == 6000
    best = load_surrogate(run / "checkpoints" / "best")
    assert best.stream_trained and best.n_features == 6
    assert best.metadata["step"] == report["best"]["step"]
    manifest_path = run / "checkpoints" / "best" / "model.json"
    assert json.loads(manifest_path.read_text())["stream_records"] == "phi"
    # The selected checkpoint reproduces its validation loss.
    validation = torch.as_tensor(np.load(tmp_path / "store" / "validation.npy")[:1500])
    logits = stream_logits(best, validation).double()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, event_labels(validation).double()
    )
    assert float(loss) == pytest.approx(report["best"]["loss"], rel=1e-6)
    rows = json.loads((run / "metrics.json").read_text())
    assert {(r["checkpoint"], r["fidelity"]) for r in rows} == {
        (c, f) for c in ("best", "final") for f in ("stream", "lf", "hf")
    }
    hf = next(r for r in rows if r["fidelity"] == "hf" and r["checkpoint"] == "best")
    assert hf["design_point_rate_mae"] is not None
    # predict maps design-point inputs into the record columns the trees were trained on.
    model, provider, _ = open_for_prediction(
        run / "checkpoints" / "best", tmp_path / "design_points"
    )
    batch = provider.load("gp_validation", "hf")[0]
    from core.surrogate_bdt import stream_records

    direct = model.estimator.decision_function(stream_records(batch, NORMALIZATION))
    np.testing.assert_allclose(model.predict(batch).logits.ravel(), direct)
    with pytest.raises(FileExistsError):
        run_stream(config)
    # A stream BDT written before the change (records with theta + offset) is refused.
    manifest = json.loads(manifest_path.read_text())
    manifest["stream_position"] = [0, 1, 2]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Retrain it"):
        load_surrogate(run / "checkpoints" / "best")


def test_stream_bdt_without_theta_scores_an_event_only_view(tmp_path):
    pytest.importorskip("sklearn")
    model = {"kind": "bdt", "architecture": {"max_iter": 20, "eval_every": 10, "max_depth": 2}}
    training = {"negatives_per_positive": 4, "batch_size": 1000, "validation_events": 1500}
    data = _variant(tmp_path, model, training).model_dump(mode="json")
    data.update(dim_theta=None, data_provider="tests.test_stream_field:EventOnlyFieldNpz")
    run_stream(StreamRunConfig.model_validate(data))
    run = tmp_path / "run"
    best = load_surrogate(run / "checkpoints" / "best")
    assert best.dim_theta is None and best.ignores_theta and best.n_features == 6
    rows = json.loads((run / "metrics.json").read_text())
    hf = next(r for r in rows if r["fidelity"] == "hf" and r["checkpoint"] == "best")
    assert set(hf["feature_calibration"]) == {f"phi_{j}" for j in range(6)}
    # The same trees score the view with theta to the same logits.
    from tests.test_bernoulli_field import FieldNpz

    best.configure_normalization(NORMALIZATION)
    full = FieldNpz(tmp_path / "design_points").load("gp_validation", "hf")[0]
    event_only = full.model_copy(update={"mode": InputMode.EVENT_ONLY, "theta": None})
    np.testing.assert_array_equal(best.predict(full).logits, best.predict(event_only).logits)
