"""Neural field: encodings, physical-unit input combination, persistence and pipeline."""

import json
import math

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from core.bernoulli_field import FourierFeatures, SphericalHarmonics
from core.surrogates import load_surrogate
from core.surrogates.datasets import NpzBatches
from core.surrogates.models import build_surrogate
from schemas.data_models import InputMode, StandardBatch
from schemas.surrogates import FieldModel

# theta = design point centre (3), phi = emission offset (3) + direction vector (3).
NORMALIZATION = {
    "theta": {"method": "standard", "offset": [0.5, -0.2, 1.0], "scale": [0.06, 0.05, 0.07]},
    "phi": {
        "method": "standard",
        "offset": [1e-4, -2e-4, 0.0, 0.0, 0.0, 0.0],
        "scale": [1.4e-3, 1.5e-3, 1.3e-3, 5.6e-6, 5.5e-6, 5.7e-6],
    },
}
SPEC = {
    "kind": "field",
    "architecture": {"hidden_dims": [16, 16]},
    "position": {"phi_columns": [0, 1, 2]},
    "direction": {"phi_columns": [3, 4, 5], "degree": 3},
    "encoding": {"n_frequencies": 4, "max_frequency": 4.0},
}


def _normalize(values, group):
    entry = NORMALIZATION[group]
    return (values - np.asarray(entry["offset"])) / np.asarray(entry["scale"])


def _physical_batch(centre, offset, direction, labels=None, counts=None):
    """StandardBatch from physical centre [B, 3], offset [B, N, 3], direction [B, N, 3]."""
    b, n = offset.shape[:2]
    phi = np.concatenate([offset, direction], axis=-1)
    return StandardBatch(
        mode=InputMode.FULL,
        theta=_normalize(centre, "theta"),
        phi=_normalize(phi, "phi").astype(np.float32),
        labels=np.zeros((b, n), dtype=np.int8) if labels is None else labels,
        event_counts=counts,
    )


def _field(seed=0, **changes):
    model = build_surrogate({**SPEC, **changes}, 3, 6, seed=seed)
    model.configure_normalization(NORMALIZATION)
    return model


def _random_physical(rng, b=4, n=30):
    centre = np.asarray(NORMALIZATION["theta"]["offset"]) + rng.normal(0, 0.05, (b, 3))
    offset = rng.uniform(-2.5e-3, 2.5e-3, (b, n, 3))
    direction = rng.normal(size=(b, n, 3))
    direction *= 9.7e-6 / np.linalg.norm(direction, axis=-1, keepdims=True)
    return centre, offset, direction


def test_spherical_harmonics_are_orthonormal():
    torch.manual_seed(0)
    u = torch.randn(400_000, 3, dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True)
    y = SphericalHarmonics(4)(u)  # [M, 25]
    gram = 4 * math.pi * (y.T @ y) / len(u)  # ∫ Y_a Y_b dΩ by Monte Carlo
    np.testing.assert_allclose(gram.numpy(), np.eye(25), atol=0.02)
    # Degree 1 is the (scaled) unit vector itself, ordered m = -1, 0, 1 -> y, z, x.
    c = math.sqrt(3 / (4 * math.pi))
    np.testing.assert_allclose(y[:, 1:4].numpy(), c * u[:, [1, 2, 0]].numpy(), atol=1e-12)
    # Smooth at the poles (no atan2 singularity).
    pole = SphericalHarmonics(4)(torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))
    assert torch.isfinite(pole).all()


def test_fourier_features_shape_and_frequencies():
    encoding = FourierFeatures(4, 8.0)
    np.testing.assert_allclose(encoding.frequencies.numpy(), [1.0, 2.0, 4.0, 8.0])
    out = encoding(torch.zeros(2, 5, 3))
    assert out.shape == (2, 5, encoding.output_dim(3)) == (2, 5, 27)


def test_position_is_continuous_across_design_points():
    """Same emission point from two different design point centres gives the same logit."""
    rng = np.random.default_rng(1)
    model = _field()
    centre, offset, direction = _random_physical(rng)
    shift = rng.uniform(-3e-3, 3e-3, (len(centre), 3))  # move the centre, keep the point
    a = model.predict(_physical_batch(centre, offset, direction))
    b = model.predict(_physical_batch(centre + shift, offset - shift[:, None, :], direction))
    np.testing.assert_allclose(a.logits, b.logits, atol=1e-4)
    # Moving the emission point does change it.
    c = model.predict(_physical_batch(centre + shift, offset, direction))
    assert np.abs(a.logits - c.logits).max() > 1e-4


def test_direction_length_does_not_matter():
    rng = np.random.default_rng(2)
    model = _field()
    centre, offset, direction = _random_physical(rng)
    a = model.predict(_physical_batch(centre, offset, direction))
    b = model.predict(_physical_batch(centre, offset, 3.0 * direction))
    np.testing.assert_allclose(a.logits, b.logits, atol=1e-4)


def test_checkpoint_keeps_normalization(tmp_path):
    rng = np.random.default_rng(3)
    model = _field()
    batch = _physical_batch(*_random_physical(rng))
    model.save(tmp_path / "ckpt")
    restored = load_surrogate(tmp_path / "ckpt")
    np.testing.assert_allclose(restored.predict(batch).logits, model.predict(batch).logits)
    assert not restored.uses_context


def test_ragged_padding_with_zero_directions():
    rng = np.random.default_rng(4)
    centre, offset, direction = _random_physical(rng, b=3, n=10)
    counts = np.array([10, 4, 7])
    valid = np.arange(10) < counts[:, None]
    phi = np.concatenate([offset, direction], -1)
    batch = StandardBatch(
        mode=InputMode.FULL,
        theta=_normalize(centre, "theta"),
        phi=np.where(valid[..., None], _normalize(phi, "phi"), 0.0).astype(np.float32),
        labels=np.zeros((3, 10), dtype=np.int8),
        event_counts=counts,
    )
    # Normalized zero padding maps to a physical vector with tiny offset only; force exact 0.
    model = _field()
    model.module.direction_shift.zero_()
    assert np.isfinite(model.predict(batch).mean).all()


def test_invalid_field_settings():
    with pytest.raises(ValidationError):
        FieldModel(position={"phi_columns": [0, 1, 2]}, direction={"phi_columns": [2, 3, 4]})
    with pytest.raises(ValueError, match="one to one"):
        build_surrogate({**SPEC, "position": {"phi_columns": [0, 1]}}, 3, 6)
    with pytest.raises(ValueError, match="phi_columns"):
        build_surrogate(SPEC, 3, 5)
    with pytest.raises(ValueError, match="normalization"):
        build_surrogate(SPEC, 3, 6).configure_normalization({"theta": {}, "phi": {}})
    # Without position/direction the field runs on Fourier(theta) and raw phi.
    plain = build_surrogate({"kind": "field", "architecture": {"hidden_dims": [4]}}, 1, 1)
    assert plain.module.extra_columns == [0]


class FieldNpz(NpzBatches):
    def normalization(self):
        return NORMALIZATION


@pytest.mark.slow
def test_field_runs_through_the_pipeline(tmp_path):
    from core.surrogates.experiment import run_experiment
    from core.surrogates.inference import predict_partition
    from schemas.surrogates import SurrogateRunConfig

    rng = np.random.default_rng(5)
    root = tmp_path / "data"
    for split, fidelity in [("train", "lf"), ("validation", "lf"), ("validation", "hf")]:
        centre, offset, direction = _random_physical(rng, b=5, n=40)
        p = 0.05 + 0.2 * (direction[..., 2] > 0) / np.linalg.norm(direction, axis=-1) * 9.7e-6
        batch = _physical_batch(centre, offset, direction, labels=rng.binomial(1, p))
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
        model={**SPEC, "last_layer": {"draws": 200}},
        training=dict(
            backend="neural",
            n_steps=6,
            eval_every=3,
            batch_size=2,
            n_events=12,
        ),
        data_directory=root,
        data_provider="tests.test_bernoulli_field:FieldNpz",
        output_directory=tmp_path / "run",
        validation_context_events=4,
        selection="bernoulli_log_loss",
    )
    output = run_experiment(config)
    manifest = json.loads((output / "checkpoints/best/model.json").read_text())
    assert manifest["model"]["kind"] == "field"
    restored = load_surrogate(output / "checkpoints/best")
    expected = NORMALIZATION["phi"]["scale"][0] / NORMALIZATION["theta"]["scale"][0]
    assert restored.module.position_scale[0].item() == pytest.approx(expected, rel=1e-6)
    rows = json.loads((output / "metrics.json").read_text())
    assert all("design_point_predictive" in r for r in rows if r["split"] == "validation")
    compute = json.loads((output / "compute.json").read_text())
    assert [s["name"] for s in compute["stages"]] == [
        "data_preparation",
        "data_loading",
        "training",
        "last_layer_fit",
        "evaluation",
    ]
    assert compute["total"]["core_hours"] > 0
    path = predict_partition(output / "checkpoints/best", root, "validation", "hf")
    with np.load(path) as arrays:
        assert arrays["probability_draws"].shape == (200, 5)
    provenance = json.loads(path.with_suffix(".json").read_text())
    assert [s["name"] for s in provenance["compute"]["stages"]] == ["data_loading", "prediction"]
