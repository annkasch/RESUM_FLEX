"""Per-event logits from context-free inference: float32 shards of whole voxels + index."""

import json

import numpy as np
import pytest

from core.surrogates.inference import predict_partition, read_event_logits, write_event_logits


def _logits(counts, n, seed=0):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    logits = rng.normal(size=(len(counts), n))  # [B, N]
    logits[np.arange(n) >= np.asarray(counts)[:, None]] = 99.0  # padding, must be dropped
    return logits


def _valid(logits, counts):
    return np.concatenate([row[:c] for row, c in zip(logits, counts, strict=True)])


def test_round_trip_keeps_event_order_and_drops_padding(tmp_path):
    counts = np.array([3, 5, 1, 4, 2])
    logits, ids = _logits(counts, 6), np.array([10, 11, 12, 13, 14])
    write_event_logits(tmp_path / "events", logits, counts, ids, shard_events=6)
    read_ids, read_counts, flat = read_event_logits(tmp_path / "events")
    np.testing.assert_array_equal(read_ids, ids)
    np.testing.assert_array_equal(read_counts, counts)
    assert flat.dtype == np.float32
    np.testing.assert_allclose(flat, _valid(logits, counts).astype(np.float32))
    index = json.loads((tmp_path / "events/index.json").read_text())
    # Whole voxels per shard, at most 6 events unless one voxel alone exceeds it.
    assert [s["voxels"] for s in index["shards"]] == [[0, 1], [1, 3], [3, 5]]
    assert index["events"] == counts.sum() and index["voxels"] == 5


def test_voxel_larger_than_shard_gets_its_own_shard(tmp_path):
    counts = np.array([2, 9, 2])
    write_event_logits(tmp_path / "e", _logits(counts, 9), counts, np.arange(3), shard_events=4)
    shards = json.loads((tmp_path / "e/index.json").read_text())["shards"]
    assert [s["voxels"] for s in shards] == [[0, 1], [1, 2], [2, 3]]


def test_voxel_range_reads_only_that_range(tmp_path):
    counts = np.array([3, 5, 1, 4, 2])
    logits = _logits(counts, 6)
    write_event_logits(tmp_path / "e", logits, counts, np.arange(5), shard_events=6)
    ids, read_counts, flat = read_event_logits(tmp_path / "e", voxels=(2, 4))
    np.testing.assert_array_equal(ids, [2, 3])
    np.testing.assert_array_equal(read_counts, [1, 4])
    np.testing.assert_allclose(flat, _valid(logits[2:4], counts[2:4]).astype(np.float32))
    assert read_event_logits(tmp_path / "e", voxels=(3, 3))[2].size == 0
    with pytest.raises(ValueError, match="outside"):
        read_event_logits(tmp_path / "e", voxels=(0, 6))


def test_never_overwrites(tmp_path):
    counts = np.array([2])
    write_event_logits(tmp_path / "e", _logits(counts, 2), counts, [0])
    with pytest.raises(FileExistsError):
        write_event_logits(tmp_path / "e", _logits(counts, 2), counts, [0])


def test_rejects_donor_events_before_loading_anything(tmp_path):
    with pytest.raises(ValueError, match="events_from"):
        predict_partition(
            tmp_path / "missing", tmp_path, "test", events_from=("a", "lf"), event_logits=True
        )
