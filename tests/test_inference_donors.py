"""Donor event inputs for event-model inference (a fresh emission sample per design point)."""

import json

import numpy as np
import pytest

from core.surrogates.inference import _donor_batch
from schemas.data_models import InputMode, StandardBatch


def _batch(b, n, offset, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    counts = rng.integers(n // 2, n + 1, size=b)
    return StandardBatch(
        mode=InputMode.FULL,
        theta=offset + rng.normal(size=(b, 3)),
        phi=rng.normal(size=(b, n, 6)).astype(np.float32),
        labels=np.zeros((b, n), dtype=np.int8),
        event_counts=counts,
    )


def test_donor_batch_keeps_theta_and_takes_donor_events():
    target, donor = _batch(5, 10, 100.0, 1), _batch(7, 12, -100.0, 2)
    batch, index = _donor_batch(target, donor, seed=3)
    np.testing.assert_array_equal(batch.theta, target.theta)
    np.testing.assert_array_equal(batch.phi, donor.phi[index])
    np.testing.assert_array_equal(
        batch.events_per_design_point, donor.events_per_design_point[index]
    )
    again, index_again = _donor_batch(target, donor, seed=3)
    np.testing.assert_array_equal(index, index_again)  # seeded
    assert not np.array_equal(index, _donor_batch(target, donor, seed=4)[1])


def test_donors_never_include_the_target_itself():
    from core.surrogates.inference import draw_donors

    ids = [f"p{i}" for i in range(4)]
    for seed in range(50):
        index = draw_donors(4, ids, ids, seed)
        assert np.all(index != np.arange(4)) and np.all((0 <= index) & (index < 4))
    # Every other design point is drawn: uniform over the remaining three.
    drawn = np.stack([draw_donors(4, ids, ids, seed) for seed in range(400)])  # [S, 4]
    for target in range(4):
        counts = np.bincount(drawn[:, target], minlength=4)
        assert counts[target] == 0 and counts.min(initial=400, where=np.arange(4) != target) > 80
    # A pool of other design points keeps the earlier unrestricted draw.
    others = [f"q{i}" for i in range(7)]
    np.testing.assert_array_equal(
        draw_donors(5, others, [f"t{i}" for i in range(5)], 3),
        np.random.default_rng(np.random.SeedSequence([3, 0])).integers(7, size=5),
    )
    with pytest.raises(ValueError, match="another design point"):
        draw_donors(1, ["a"], ["a"], 0)
    with pytest.raises(ValueError, match="not unique"):
        draw_donors(1, ["a", "a"], ["b"], 0)


def test_predict_with_donors_from_the_same_partition_uses_the_provider_transfer(tmp_path):
    from core.surrogates import build_surrogate
    from core.surrogates.checkpoints import save_surrogate
    from core.surrogates.inference import predict_partition

    root = tmp_path / "data"
    target = _batch(6, 10, 0.0, 1)
    np.savez(
        root.mkdir() or root / "arrays.npz",
        theta=target.theta,
        phi=target.phi,
        counts=target.events_per_design_point,
    )
    model = build_surrogate({"kind": "mlp", "architecture": {"hidden_dims": [8]}}, 3, 6, seed=0)
    model.metadata = {
        "normalization.json": NORMALIZATION,
        "step": 1,
        "data_provider": "tests.test_inference_donors:ShiftProvider",
    }
    checkpoint = tmp_path / "run" / "checkpoints" / "best"
    save_surrogate(checkpoint, model)
    ShiftProvider.calls.clear()
    path = predict_partition(checkpoint, root, "test", "lf", events_from=("test", "lf"), seed=2)
    record = json.loads(path.with_suffix(".json").read_text())["event_inputs"]
    index = np.array(record["donor_index"])
    assert np.all(index != np.arange(6)) and record["own_excluded"]
    assert record["transfer"] == "provider transfer_events"
    ((phi, source, rows, destination),) = ShiftProvider.calls
    assert source == destination == ("test", "lf")
    np.testing.assert_array_equal(rows, index)
    np.testing.assert_array_equal(phi, target.phi[index])
    # The prediction used the moved events: equal to predicting the shifted batch directly.
    moved = StandardBatch(
        mode=InputMode.FULL,
        theta=target.theta,
        phi=phi + 1.0,
        labels=np.zeros(phi.shape[:2], dtype=np.int8),
        event_counts=target.events_per_design_point[index],
    )
    with np.load(path) as arrays:
        np.testing.assert_allclose(arrays["mean_probability"], model.predict(moved).mean, rtol=1e-6)


NORMALIZATION = {"theta": {"method": "none"}, "phi": {"method": "none"}}  # recorded as equal


def test_donors_need_the_provider_transfer_rule(tmp_path):
    from core.surrogates import build_surrogate
    from core.surrogates.checkpoints import save_surrogate
    from core.surrogates.inference import predict_partition

    root = tmp_path / "data"
    target = _batch(3, 4, 0.0, 1)
    root.mkdir()
    np.savez(
        root / "arrays.npz",
        theta=target.theta,
        phi=target.phi,
        counts=target.events_per_design_point,
    )
    model = build_surrogate({"kind": "mlp", "architecture": {"hidden_dims": [4]}}, 3, 6, seed=0)
    model.metadata = {
        "normalization.json": NORMALIZATION,
        "step": 1,
        "data_provider": "tests.test_inference_donors:PlainProvider",
    }
    checkpoint = tmp_path / "run" / "checkpoints" / "best"
    save_surrogate(checkpoint, model)
    with pytest.raises(ValueError, match="transfer_events"):
        predict_partition(checkpoint, root, "test", "lf", events_from=("test", "lf"))
    path = predict_partition(checkpoint, root, "test", "lf")  # default: own events
    assert json.loads(path.with_suffix(".json").read_text())["event_inputs"] == "own"


class ShiftProvider:
    """Minimal provider over arrays.npz whose transfer adds 1 to every donor event input."""

    calls = []

    def __init__(self, root):
        self.root = root
        with np.load(root / "arrays.npz") as arrays:
            self.arrays = {k: arrays[k] for k in arrays}

    def partitions(self):
        return ["test"]

    def load(self, partition, fidelity, *, labels=True):
        a = self.arrays
        batch = StandardBatch(
            mode=InputMode.FULL,
            theta=a["theta"],
            phi=a["phi"],
            labels=np.zeros(a["phi"].shape[:2], dtype=np.int8),
            event_counts=a["counts"],
        )
        return batch, {"partition": partition}

    def theta(self, partition, fidelity):
        return self.arrays["theta"], [f"d{i}" for i in range(len(self.arrays["theta"]))]

    def coordinates(self, partition, fidelity):
        return self.arrays["theta"]

    def normalization(self):
        return NORMALIZATION

    def transfer_events(self, phi, source, source_rows, target):
        ShiftProvider.calls.append((np.array(phi), source, np.array(source_rows), target))
        return phi + 1.0


class PlainProvider(ShiftProvider):
    """The same provider without a transfer rule."""

    transfer_events = None
