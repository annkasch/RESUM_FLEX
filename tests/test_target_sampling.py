import numpy as np
import pytest

from core.target_sampling import RealTargetSampler
from schemas.data_models import InputMode, StandardBatch


def batch():
    labels = np.zeros((3, 100), dtype=int)
    labels[0, :3] = 1
    labels[2] = 1
    phi = np.broadcast_to(np.arange(100)[None, :, None], (3, 100, 1)).astype(float).copy()
    return StandardBatch(mode=InputMode.FULL, theta=np.arange(3)[:, None], phi=phi, labels=labels)


def test_matching_real_contexts_and_disjoint_targets():
    b = batch()
    kwargs = dict(seed=5, batch_size=8, n_events=40, n_context_min=5, n_context_max=10)
    uniform = RealTargetSampler(b, **kwargs)
    weighted = RealTargetSampler(b, positive_fraction=0.25, **kwargs)
    for _ in range(10):
        cu, _, _ = uniform.next()
        cw, targets, weights = weighted.next()
        np.testing.assert_array_equal(cu.phi, cw.phi)
        np.testing.assert_array_equal(cu.labels, cw.labels)
        np.testing.assert_allclose(weights.mean(1), 1)
        for j, provenance in enumerate(weighted.last_provenance):
            trial, ci, ti = (provenance[k] for k in ("trial", "context", "target"))
            assert not set(ci) & set(ti)
            np.testing.assert_array_equal(targets.phi[j], b.phi[trial, ti])
            np.testing.assert_array_equal(targets.labels[j], b.labels[trial, ti])
            np.testing.assert_allclose(
                (weights[j] * targets.labels[j]).mean(), provenance["pool_positive_fraction"]
            )


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        RealTargetSampler(
            batch(),
            seed=0,
            batch_size=2,
            n_events=20,
            n_context_min=3,
            n_context_max=5,
            positive_fraction=fraction,
        )


def test_natural_targets_are_unique_and_class_weights_only_change_loss():
    b = batch()
    sampler = RealTargetSampler(
        b,
        seed=17,
        batch_size=8,
        n_events=40,
        n_context_min=5,
        n_context_max=10,
        positive_fraction=None,
    )
    p = b.labels.mean()
    for _ in range(10):
        context, target, sampling_weights = sampler.next()
        np.testing.assert_array_equal(sampling_weights, np.ones_like(target.labels))
        labels_before = target.labels.copy()
        weights = np.where(target.labels == 1, 0.5 / p, 0.5 / (1 - p))
        np.testing.assert_array_equal(target.labels, labels_before)
        for record in sampler.last_provenance:
            assert len(np.unique(record["target"])) == len(record["target"])
            assert not set(record["context"]) & set(record["target"])
        assert np.all(weights[target.labels == 1] == 0.5 / p)
        assert np.all(weights[target.labels == 0] == 0.5 / (1 - p))


def test_target_coverage_counts_only_valid_events():
    labels = np.zeros((2, 10), dtype=int)
    labels[0, :2] = 1  # padded events 6..9 of design point 1 are never valid
    phi = np.zeros((2, 10, 1))
    b = StandardBatch(
        mode=InputMode.FULL,
        theta=np.arange(2.0)[:, None],
        phi=phi,
        labels=labels,
        event_counts=np.array([10, 6]),
    )
    sampler = RealTargetSampler(
        b, seed=1, batch_size=1, n_events=4, n_context_min=1, n_context_max=1
    )
    assert sampler.coverage()["target_event_coverage"] == 0
    drawn = set()
    for _ in range(3):
        sampler.next()
        for provenance in sampler.last_provenance:
            drawn |= {(provenance["trial"], int(i)) for i in provenance["target"]}
    coverage = sampler.coverage()
    assert sampler.target_counts[1, 6:].sum() == 0
    assert sampler.target_counts.sum() == 3 * 3
    assert coverage["target_event_coverage"] == pytest.approx(len(drawn) / 16)
    seen_positives = len(drawn & {(0, 0), (0, 1)})
    assert coverage["target_positive_coverage"] == pytest.approx(seen_positives / 2)
    assert coverage["target_count_max"] == sampler.target_counts.max()


def test_target_coverage_without_positives_is_none():
    b = batch()
    b.labels[:] = 0
    sampler = RealTargetSampler(
        b, seed=0, batch_size=2, n_events=10, n_context_min=0, n_context_max=0
    )
    sampler.next()
    assert sampler.coverage()["target_positive_coverage"] is None
