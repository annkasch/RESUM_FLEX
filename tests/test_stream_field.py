"""Neural field trained on ungrouped events: units, design point scoring, training run, schema."""

import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from core.event_stream import event_labels
from core.surrogates import load_surrogate
from core.surrogates.models import build_surrogate
from core.surrogates.stream_training import run_stream
from schemas.data_models import InputMode, StandardBatch
from schemas.surrogates import FieldPosition, StreamRunConfig
from tests.test_bernoulli_field import (
    NORMALIZATION,
    SPEC,
    _field,
    _physical_batch,
    _random_physical,
)

# Store normalization of x, y, z (m) and px, py, pz (MeV), deliberately unlike the design point one.
STREAM_NORMALIZATION = {
    "offset": [0.45, -0.21, 1.02, 0.0, 0.0, 0.0],
    "scale": [0.08, 0.04, 0.09, 5.6e-6, 5.6e-6, 5.6e-6],
}


# Stream MLP: raw position and momentum (what ``model: {kind: mlp}`` trains on events).
MLP_CHANGES = {
    "encoding": {"kind": "none"},
    "direction": {"phi_columns": [3, 4, 5], "encoding": "momentum"},
}


def _records(position, momentum, label=None):
    """Raw event records [M, 7] float32 with label bits in the last column."""
    out = np.zeros((len(position), 7), dtype=np.float32)
    out[:, :3], out[:, 3:6] = position, momentum
    bits = np.zeros(len(position), dtype=np.int32) if label is None else label.astype(np.int32)
    out[:, 6] = bits.view(np.float32)
    return out


class FakeEventSource:
    """In-memory event source over ``<store>/<split>.npy`` records (see core.event_stream)."""

    opened_with: list = []  # the options of every source opened (event_source_options)

    def __init__(self, store, split, *, seed=0, rank=0, world_size=1, **options):
        FakeEventSource.opened_with.append(options)
        self.records = np.load(f"{store}/{split}.npy")
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.n_events = len(self.records)
        self.n_positive = int(event_labels(torch.as_tensor(self.records)).sum())

    def normalization(self):
        return STREAM_NORMALIZATION

    def n_batches(self, batch_size, pass_index, *, drop_last=False):
        n = len(self.records) // self.world_size
        return n // batch_size if drop_last else -(-n // batch_size)

    def batches(self, batch_size, pass_index=0, *, start_batch=0, drop_last=False, device="cpu"):
        order = np.random.default_rng([self.seed, pass_index]).permutation(len(self.records))
        order = order[self.rank :: self.world_size]
        for b in range(start_batch, self.n_batches(batch_size, pass_index, drop_last=drop_last)):
            chunk = self.records[order[b * batch_size : (b + 1) * batch_size]]
            yield torch.as_tensor(chunk, device=device)

    def load(self):
        return self.records


@pytest.mark.parametrize("changes", [{}, MLP_CHANGES], ids=["field", "mlp"])
def test_stream_trained_field_gives_the_same_logits_on_design_point_inputs(changes):
    model = _field(**changes)
    model.module.set_stream_normalization(STREAM_NORMALIZATION)
    model.configure_normalization(NORMALIZATION)  # design point data of the evaluation
    rng = np.random.default_rng(1)
    centre, offset, direction = _random_physical(rng)
    direction *= rng.uniform(0.9, 1.1, direction.shape[:-1])[..., None]  # wavelength spread
    batch = _physical_batch(centre, offset, direction)
    with torch.no_grad():
        design_point = model.module(None, batch)  # [B, N]
        position = (centre[:, None, :] + offset).reshape(-1, 3)
        events = torch.as_tensor(_records(position, direction.reshape(-1, 3)))
        direct = model.module.forward_events(events).reshape(design_point.shape)
    torch.testing.assert_close(design_point, direct, rtol=1e-4, atol=1e-4)


def test_stream_mlp_sees_the_momentum_length_and_the_field_does_not():
    rng = np.random.default_rng(3)
    position = rng.normal(STREAM_NORMALIZATION["offset"][:3], 0.05, (50, 3))
    momentum = rng.normal(size=(50, 3)) * 9.7e-6
    outputs = {}
    for name, changes in (("field", {}), ("mlp", MLP_CHANGES)):
        model = _field(**changes)
        model.module.set_stream_normalization(STREAM_NORMALIZATION)
        with torch.no_grad():
            short, long = (
                model.module.forward_events(torch.as_tensor(_records(position, k * momentum)))
                for k in (1.0, 1.05)
            )
        outputs[name] = float((short - long).abs().max())
    assert outputs["field"] < 1e-5 < outputs["mlp"]


def test_stream_mlp_config_is_the_unencoded_field():
    config = StreamRunConfig.model_validate(
        {
            "model": {"kind": "mlp", "architecture": {"hidden_dims": [32, 32]}},
            "event_store": "s",
            "event_source": "m:C",
            "output_directory": "o",
            "dim_theta": 3,
            "dim_phi": 6,
            "mlp_position": {"phi_columns": [0, 1, 2]},
        }
    )
    network = config.network
    assert network.kind == "field" and network.encoding.kind == "none"
    assert network.direction is None
    assert network.position.phi_columns == [0, 1, 2]
    assert list(network.architecture.hidden_dims) == [32, 32]


def test_design_point_trained_field_is_unchanged_and_old_weights_load():
    model = _field()
    rng = np.random.default_rng(2)
    batch = _physical_batch(*_random_physical(rng))
    state = model.module.state_dict()
    old = {k: v for k, v in state.items() if k not in model.module._OPTIONAL_BUFFERS}
    restored = _field(seed=5)
    restored.module.load_state_dict(old)
    with torch.no_grad():
        torch.testing.assert_close(restored.module(None, batch), model.module(None, batch))
    assert not restored.module.uses_stream_units
    with pytest.raises(ValueError, match="set_stream_normalization"):
        model.module.forward_events(torch.zeros((2, 7)))


def _stream_split(rng, n):
    """Events in a box with p(detected) rising along x (a learnable rate)."""
    offset = np.asarray(STREAM_NORMALIZATION["offset"][:3])
    position = offset + rng.uniform(-0.1, 0.1, (n, 3))
    momentum = rng.normal(size=(n, 3)) * 9.7e-6
    p = 1 / (1 + np.exp(-(-3.0 + 25.0 * (position[:, 0] - offset[0]))))
    return _records(position, momentum, rng.random(n) < p)


def _design_point_data(root, rng):
    """Design point evaluation data (NpzBatches layout) with the same rate as the events."""
    x0 = STREAM_NORMALIZATION["offset"][0]
    for partition, fidelity in (("cnp_validation", "lf"), ("gp_validation", "hf")):
        centre, offset, direction = _random_physical(rng, b=6, n=60)
        x = centre[:, None, 0] + offset[..., 0]
        p = 1 / (1 + np.exp(-(-3.0 + 25.0 * (x - x0))))
        batch = _physical_batch(centre, offset, direction, labels=rng.binomial(1, p))
        folder = root / "batches" / partition
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{fidelity}.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )


def _run_config(tmp_path, **training):
    store = tmp_path / "store"
    store.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 6000), ("validation", 2000)):
        np.save(store / f"{split}.npy", _stream_split(rng, n))
    _design_point_data(tmp_path / "design_points", rng)
    return StreamRunConfig.model_validate(
        {
            "model": {**SPEC, "last_layer": {"prior_precision": "evidence", "draws": 100}},
            "training": {
                "batch_size": 256,
                "n_passes": 3,
                "eval_every": 20,
                "learning_rate": 3e-3,
                "device": "cpu",
                "validation_events": 1500,
                **training,
            },
            "event_store": str(store),
            "event_source": "tests.test_stream_field:FakeEventSource",
            "output_directory": str(tmp_path / "run"),
            "data_provider": "tests.test_bernoulli_field:FieldNpz",
            "data_directory": str(tmp_path / "design_points"),
            "evaluation": [
                {"split": "validation", "partition": "cnp_validation", "fidelity": "lf"},
                {"split": "validation", "partition": "gp_validation", "fidelity": "hf"},
            ],
            "history_design_points": 4,
            "dim_theta": 3,
            "dim_phi": 6,
        }
    )


def test_stream_training_learns_selects_and_fits_the_last_layer(tmp_path):
    config = _run_config(tmp_path)
    report = run_stream(config)
    history = report["history"]
    assert report["steps"] == 3 * -(-6000 // 256)
    assert history[-1]["stream_log_loss"] < history[0]["stream_log_loss"]
    # Every validation step also scores the (subsampled) LF validation design points.
    assert all(r["design_points"] == 4 and "design_point_rate_mae" in r for r in history)
    run = tmp_path / "run"
    experiment = json.loads((run / "experiment.json").read_text())
    assert experiment["stream_normalization"] == STREAM_NORMALIZATION
    assert experiment["validation_events"] == 1500
    assert (run / "compute.json").is_file()
    best = load_surrogate(run / "checkpoints" / "best")
    assert best.module.uses_stream_units and best.last_layer is not None
    assert best.metadata["last_layer"]["events"] == 6000
    assert best.metadata["best_stream_log_loss"] == pytest.approx(report["best"]["loss"])
    assert best.metadata["step"] == report["best"]["step"]
    # Same metrics and plots as design point training, plus the stream ones.
    rows = json.loads((run / "metrics.json").read_text())
    keys = {(r["checkpoint"], r["split"], r["fidelity"]) for r in rows}
    assert keys == {(c, "validation", f) for c in ("best", "final") for f in ("stream", "lf", "hf")}
    design_point = next(r for r in rows if r["fidelity"] == "hf" and r["checkpoint"] == "best")
    for key in (
        "design_point_rate_mae",
        "design_point_count_nll",
        "log_loss_skill",
        "design_point_predictive",
    ):
        assert design_point[key] is not None
    events = next(r for r in rows if r["fidelity"] == "stream" and r["checkpoint"] == "best")
    assert events["bernoulli_log_loss"] == pytest.approx(report["best"]["loss"], rel=1e-5)
    for name in (
        "validation_history",
        "lf_validation_zscore",
        "lf_calibration",
        "hf_precision_recall_log",
        "stream_history",
        "stream_calibration",
        "stream_precision_recall_log",
    ):
        assert (run / f"{name}.png").is_file(), name
    assert (run / "hf_validation_best_predictive_coverage.png").is_file()
    # The selected checkpoint reproduces its recorded validation loss.
    validation = torch.as_tensor(np.load(tmp_path / "store" / "validation.npy")[:1500])
    with torch.no_grad():
        logits = best.module.forward_events(validation)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, event_labels(validation))
    assert float(loss) == pytest.approx(report["best"]["loss"], rel=1e-5)
    with pytest.raises(FileExistsError):
        run_stream(config)


def test_stream_mlp_trains_and_scores_on_design_points(tmp_path):
    config = _run_config(tmp_path, n_passes=1)
    config = config.model_copy(
        update={
            "model": StreamRunConfig.model_validate(
                {
                    "model": {
                        "kind": "mlp",
                        "architecture": {"hidden_dims": [16, 16]},
                        "last_layer": {"prior_precision": "evidence", "draws": 100},
                    },
                    "event_store": "s",
                    "event_source": "tests.test_stream_field:FakeEventSource",
                    "output_directory": "o",
                    "dim_theta": 3,
                    "dim_phi": 6,
                }
            ).model,
            "mlp_position": FieldPosition(phi_columns=[0, 1, 2]),
        }
    )
    report = run_stream(config)
    assert report["history"][-1]["stream_log_loss"] < report["history"][0]["stream_log_loss"]
    best = load_surrogate(tmp_path / "run" / "checkpoints" / "best")
    assert best.module.fourier is None and best.module.direction_columns == []
    assert best.module.extra_columns == [3, 4, 5]
    assert best.last_layer is not None
    experiment = json.loads((tmp_path / "run" / "experiment.json").read_text())
    assert experiment["resolved_config"]["model"]["kind"] == "mlp"
    rows = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert {r["fidelity"] for r in rows} == {"stream", "lf", "hf"}


def test_stream_config_checks_the_columns():
    base = {"event_store": "s", "event_source": "m:C", "output_directory": "o", "dim_theta": 3}
    StreamRunConfig.model_validate({"model": SPEC, "dim_phi": 6, **base})
    with pytest.raises(ValidationError, match="match dim_theta"):
        StreamRunConfig.model_validate({"model": SPEC, "dim_phi": 6, **{**base, "dim_theta": 2}})
    with pytest.raises(ValidationError, match="dim_phi > 5"):
        StreamRunConfig.model_validate({"model": SPEC, "dim_phi": 5, **base})
    with pytest.raises(ValidationError, match="mlp_position is for kind: mlp"):
        StreamRunConfig.model_validate(
            {"model": SPEC, "dim_phi": 6, "mlp_position": {"phi_columns": [0, 1, 2]}, **base}
        )


def test_event_source_options_reach_the_source_unchanged(tmp_path):
    config = _run_config(tmp_path, n_passes=1)
    config.event_source_options = {"in_memory": True, "block": 7}
    FakeEventSource.opened_with.clear()
    run_stream(config)
    assert FakeEventSource.opened_with
    assert all(o == {"in_memory": True, "block": 7} for o in FakeEventSource.opened_with)


def test_event_source_is_required():
    with pytest.raises(ValidationError, match="event_source"):
        StreamRunConfig.model_validate(
            {
                "model": SPEC,
                "event_store": "s",
                "output_directory": "o",
                "dim_theta": 3,
                "dim_phi": 6,
            }
        )


def test_field_checkpoint_with_the_old_buffer_name_loads():
    model = _field()
    model.module.set_stream_normalization(STREAM_NORMALIZATION)
    state = model.module.state_dict()
    state["photon_units"] = state.pop("stream_units")  # name before 2026-11-06
    fresh = _field()
    fresh.module.load_state_dict(state)
    assert fresh.module.uses_stream_units


@pytest.mark.parametrize(
    "dims, model, position",
    [
        ((2, 3), {"kind": "mlp"}, [0, 1]),  # 2-D position (theta + offset) and one more input
        ((2, 2), {"kind": "mlp"}, None),  # theta alone is the position, both phi columns extra
        ((1, 5), {"kind": "mlp"}, [4]),  # position offset in the last phi column
        (
            (2, 6),  # field: 2-D position, 3-D direction, one more input
            {**SPEC, "position": {"phi_columns": [0, 1]}, "direction": {"phi_columns": [2, 3, 4]}},
            None,
        ),
    ],
    ids=["mlp_2+1", "mlp_theta_only", "mlp_1+4", "field_2+3+1"],
)
def test_stream_models_take_other_record_layouts(dims, model, position):
    """Logits from stream records equal those from the same events as design-point data."""
    dim_theta, dim_phi = dims
    config = StreamRunConfig.model_validate(
        {
            "model": {"architecture": {"hidden_dims": [16, 16]}, **model},
            "event_store": "s",
            "event_source": "m:C",
            "output_directory": "o",
            "dim_theta": dim_theta,
            "dim_phi": dim_phi,
            **({"mlp_position": {"phi_columns": position}} if position else {}),
        }
    )
    network = config.network
    rng = np.random.default_rng(4)
    b, n = 3, 20
    theta = rng.normal(5.0, 2.0, (b, dim_theta))  # physical units
    phi = rng.normal(-1.0, 0.5, (b, n, dim_phi))
    pos = [] if network.position is None else network.position.phi_columns
    rest = [c for c in range(dim_phi) if c not in pos]
    record_position = theta[:, None, :] + (phi[..., pos] if pos else np.zeros((b, n, 1)))
    records = np.concatenate([record_position, phi[..., rest]], axis=-1).reshape(b * n, -1)
    data_norm = {
        "theta": {"offset": [4.0] * dim_theta, "scale": [3.0] * dim_theta},
        "phi": {"offset": [-0.5] * dim_phi, "scale": [0.7] * dim_phi},
    }
    width = records.shape[1]
    stream_norm = {"offset": list(records.mean(0)), "scale": list(records.std(0))}
    surrogate = build_surrogate(network, dim_theta, dim_phi, seed=0)
    surrogate.module.set_stream_normalization(stream_norm)
    surrogate.configure_normalization(data_norm)

    def normalized(values, group):
        entry = data_norm[group]
        return (values - np.asarray(entry["offset"])) / np.asarray(entry["scale"])

    batch = StandardBatch(
        mode=InputMode.FULL,
        theta=normalized(theta, "theta"),
        phi=normalized(phi, "phi").astype(np.float32),
        labels=np.zeros((b, n), dtype=np.int8),
    )
    labels = np.zeros((b * n, 1), dtype=np.float32)
    stream = torch.as_tensor(np.concatenate([records, labels], axis=1), dtype=torch.float32)
    assert surrogate.module.record_width == width
    with torch.no_grad():
        design = surrogate.module(None, batch)  # [B, N]
        direct = surrogate.module.forward_events(stream).reshape(b, n)
    torch.testing.assert_close(design, direct, rtol=1e-4, atol=1e-4)
