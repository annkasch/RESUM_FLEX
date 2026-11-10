"""Donor event inputs for event-model inference (a fresh emission sample per design point)."""

import numpy as np

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
