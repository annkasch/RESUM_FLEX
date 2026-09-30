import numpy as np
import pytest

from core.mixup import ClassAwareMixupSource
from schemas.data_models import InputMode, StandardBatch


def make_batch(labels):
    labels = np.asarray(labels)
    return StandardBatch(
        mode=InputMode.FULL,
        theta=np.arange(len(labels))[:, None],
        phi=labels[..., None].astype(float),
        labels=labels,
    )


def assert_disjoint(source):
    for row in range(source.batch_size):
        context, target = [r for r in source.last_provenance if r["row"] == row]
        assert context["voxel"] == target["voxel"]
        cp, tp = set(context["source_pool"]), set(target["source_pool"])
        assert not cp & tp
        assert cp | tp == set(range(source.batch.n_events))
        for record, pool in ((context, cp), (target, tp)):
            for key in ("anchors", "partners", "real_indices"):
                if key in record:
                    assert set(record[key]) <= pool


def test_fresh_splits_pairing_reproducibility_and_single_positive_changes_roles():
    batch = make_batch([[0] * 24 + [1]])
    options = dict(alpha=0.2, seed=42, batch_size=2, n_events=8, n_context=3)
    source = ClassAwareMixupSource(batch, **options)
    same = ClassAwareMixupSource(batch, **options)
    positive_sides, context_pools = set(), set()
    for _ in range(30):
        outputs = source.next()
        repeated = same.next()
        assert_disjoint(source)
        for actual, expected, size in zip(outputs, repeated, (3, 5), strict=True):
            assert actual.labels.shape == (2, size)
            np.testing.assert_array_equal(actual.labels, expected.labels)
            np.testing.assert_allclose(actual.phi[..., 0], actual.labels)
        for record in source.last_provenance:
            if record["side"] == 0:
                context_pools.add(tuple(sorted(record["source_pool"])))
            if 24 in record["source_pool"]:
                positive_sides.add(record["side"])
                assert np.all(batch.labels[0, record["anchors"]] == 0)
                assert np.all(batch.labels[0, record["partners"]] == 1)
                assert np.ptp(record["weights"]) > 0
            else:
                np.testing.assert_array_equal(record["weights"], 0)
    assert positive_sides == {0, 1}
    assert len(context_pools) > 1


def test_real_context_preserves_events_and_matched_targets():
    y = np.zeros((3, 100), dtype=int)
    y[0, -10:] = 1
    y[1, -1] = 1
    batch = make_batch(y)
    batch.phi = np.broadcast_to(np.arange(100)[None, :, None], (3, 100, 1)).astype(float).copy()
    options = dict(alpha=0.2, seed=42, batch_size=2, n_events=16, n_context=6)
    mixed = ClassAwareMixupSource(batch, **options)
    real = ClassAwareMixupSource(batch, mix_context=False, **options)
    for _ in range(30):
        _, mixed_target = mixed.next()
        context, target = real.next()
        assert_disjoint(real)
        for field in ("labels", "phi", "theta"):
            np.testing.assert_array_equal(getattr(target, field), getattr(mixed_target, field))
        assert type(context) is StandardBatch
        for a, b in zip(real.last_provenance, mixed.last_provenance, strict=True):
            np.testing.assert_array_equal(a["source_pool"], b["source_pool"])
        for r in real.last_provenance:
            if r["side"] == 0:
                indices = r["real_indices"]
                assert len(set(indices)) == len(indices)
                np.testing.assert_array_equal(context.phi[r["row"], :, 0], indices)
                np.testing.assert_array_equal(context.labels[r["row"]], y[r["voxel"], indices])


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_small_pools_return_real_events(label):
    source = ClassAwareMixupSource(
        make_batch([[label, label]]),
        alpha=0.2,
        seed=0,
        batch_size=2,
        n_events=8,
        n_context=3,
    )
    for output in source.next():
        np.testing.assert_array_equal(output.labels, label)
        np.testing.assert_array_equal(output.phi[..., 0], label)
    assert_disjoint(source)


@pytest.mark.parametrize("mix_context", [False, True])
@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
def test_real_branch_uses_target_sources_without_changing_mixtures(mix_context, ratio):
    batch = make_batch([[0] * 20 + [1] * 4])
    batch.phi = np.arange(24, dtype=float)[None, :, None]
    options = dict(
        alpha=0.2, seed=5, batch_size=2, n_events=12, n_context=4, mix_context=mix_context
    )
    plain = ClassAwareMixupSource(batch, **options)
    combined = ClassAwareMixupSource(batch, **options)
    for _ in range(5):
        pc, pt = plain.next()
        context, target, real = combined.next(real_target_ratio=ratio)
        assert real.n_events == max(1, round(8 * ratio))
        for original, augmented in ((pc, context), (pt, target)):
            np.testing.assert_array_equal(original.phi, augmented.phi)
            np.testing.assert_array_equal(original.labels, augmented.labels)
        for record in combined.last_provenance:
            if record["side"] == 1:
                idx = record["real_target_indices"]
                assert set(idx) <= set(record["source_pool"])
                np.testing.assert_array_equal(real.phi[record["row"], :, 0], idx)
                np.testing.assert_array_equal(real.labels[record["row"]], batch.labels[0, idx])
        assert_disjoint(combined)
