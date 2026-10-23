"""Shared event cache and per-config views on synthetic LH5 fixtures."""

import os

import numpy as np
import pytest

from data.event_cache import EventCache, cached_runs, main
from data.optical_features import AffineTransform, RunningMoments, make_features, make_labels
from data.optical_reader import read_run
from data.optical_view import OpticalView, build_batch
from tests.test_optical_pipeline import config, dataset, fixture_file

h5py = pytest.importorskip("h5py")


def files(tmp_path, number=6):
    root = tmp_path / "source"
    return [
        fixture_file(
            root,
            run=i,
            center=(0.1 * i, 0.2, 0.3),
            n=5 + i,
            hit_ids=(0, 2, 2, 4) if i % 2 else (),
            detectors=(7, 7, 9, 9) if i % 2 else (),
        )
        for i in range(number)
    ]


def test_cached_runs_equal_lh5_runs(tmp_path):
    c = config(tmp_path)
    paths = files(tmp_path)
    cache = EventCache(tmp_path / "cache")
    assert cache.update(paths, workers=3, min_files_per_task=2) == len(paths)
    assert len(cache.shards) == 3  # tasks of two files, one shard each
    runs, _ = cached_runs(c.source, cache)
    for run in runs:
        raw = read_run(c.source.directory / run.file, c.source.directory)
        assert run.fingerprint == raw.fingerprint and run.n_events == raw.n_events
        np.testing.assert_array_equal(run.positions, raw.positions.astype(np.float32))
        np.testing.assert_array_equal(run.nominal_center, raw.nominal_center)
        np.testing.assert_allclose(run.measured_center, raw.measured_center, atol=1e-7)
        assert run.max_n_part == 1
        # Hits refer to event rows; the fixture stores event IDs in reverse order.
        order = np.argsort(raw.event_ids)
        rows = np.searchsorted(raw.event_ids[order], raw.hit_event_ids)
        expected = sorted(zip(order[rows].tolist(), raw.detector_ids.tolist(), strict=True))
        assert (
            sorted(zip(run.hit_event_ids.tolist(), run.detector_ids.tolist(), strict=True))
            == expected
        )
        for kind in ("any_channel", "single_channel"):
            c.target.kind, c.target.detector_id = kind, 9 if kind == "single_channel" else None
            np.testing.assert_array_equal(make_labels(run, c), make_labels(raw, c))
        c.target.kind, c.target.detector_id = "any_channel", None
        theta, phi, _ = make_features(run, c)
        raw_theta, raw_phi, _ = make_features(raw, c)
        np.testing.assert_array_equal(theta, raw_theta)
        np.testing.assert_allclose(phi, raw_phi, atol=1e-7)


def test_update_reads_only_new_or_changed_files(tmp_path):
    paths = files(tmp_path, 3)
    cache = EventCache(tmp_path / "cache")
    assert cache.update(paths, workers=1) == 3
    assert cache.update(paths, workers=1) == 0
    assert EventCache(tmp_path / "cache").update(paths, workers=1) == 0  # state on disk
    with h5py.File(paths[0], "r+") as h:
        h["stp/optical/evtid"][...] = []
    os.utime(paths[0], ns=(1, 1))  # a rewrite changes size or mtime
    extra = fixture_file(tmp_path / "source", run=9, center=(2, 0, 0))
    assert cache.update([*paths, extra], workers=1) == 2
    assert cache.index["valid"].sum() == 4 and len(cache.index) == 5
    row = cache.locate([paths[0]])[0]
    assert cache.index["n_hits"][row] == 0 and cache.labels(row).sum() == 0
    with pytest.raises(ValueError, match="missing"):
        cache.locate([tmp_path / "other.stp.lh5"])
    with pytest.raises(ValueError, match="different file contents"):
        cache.locate([paths[0]], np.zeros((1, 32), dtype=np.uint8))


def test_compact_keeps_contents_and_removes_old_shards(tmp_path):
    paths = files(tmp_path)
    cache = EventCache(tmp_path / "cache")
    for path in paths:
        cache.update([path], workers=1)
    os.utime(paths[1], ns=(2, 2))
    cache.update(paths, workers=1)
    before = {
        p: (cache.events(r).copy(), cache.hits(r).copy())
        for p, r in zip(paths, cache.locate(paths), strict=True)
    }
    digests = cache.index["sha256"][cache.locate(paths)]
    main(["compact", str(tmp_path / "cache"), "--files-per-shard", "4"])
    cache = EventCache(tmp_path / "cache")
    assert len(cache.shards) == 2 and len(cache.index) == len(paths)
    assert sorted(p.name for p in (tmp_path / "cache/shards").iterdir()) == cache.shards
    rows = cache.locate(paths, digests)
    for path, row in zip(paths, rows, strict=True):
        np.testing.assert_array_equal(cache.events(row), before[path][0])
        np.testing.assert_array_equal(cache.hits(row), before[path][1])


def test_lock_blocks_a_second_writer(tmp_path):
    paths = files(tmp_path, 1)
    cache = EventCache(tmp_path / "cache")
    (tmp_path / "cache/.lock").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="locked"):
        cache.update(paths, workers=1)
    (tmp_path / "cache/.lock").rmdir()
    assert cache.update(paths, workers=1) == 1
    assert not (tmp_path / "cache/.lock").exists()


def test_running_moments_match_a_direct_fit():
    rng = np.random.default_rng(0)
    values = rng.normal(3.0, 2.0, size=(1000, 4)) * [1, 10, 1e-3, 1]
    moments = RunningMoments()
    for chunk in np.array_split(values, [1, 7, 300, 999]):
        moments.add(chunk)
    for method in ("standard", "minmax"):
        direct = AffineTransform.fit(values, method)
        merged = AffineTransform.from_moments(moments, method)
        np.testing.assert_allclose(merged.offset, direct.offset, rtol=1e-12)
        np.testing.assert_allclose(merged.scale, direct.scale, rtol=1e-12)
    np.testing.assert_allclose(merged.offset, values.min(0))
    np.testing.assert_allclose(
        AffineTransform.fit(values, "standard").scale, values.std(0), rtol=1e-12
    )


def test_view_batches_are_chunk_independent(tmp_path, monkeypatch):
    import data.optical_view as optical_view
    from data.optical_pipeline import prepare_optical_data

    c = dataset(tmp_path, 4)
    prepare_optical_data(c)
    view = OpticalView(c.output_directory)
    whole = view.load("train", "hf")
    monkeypatch.setattr(optical_view, "CHUNK_EVENTS", 13)  # about one file per chunk
    np.testing.assert_array_equal(view.load("train", "hf").phi, whole.phi)
    assert whole.phi.dtype == np.float32
    rows = view.cache.locate(view.paths(view.select("train", "hf")))
    c.target.kind, c.target.detector_id = "single_channel", 11
    single = build_batch(view.cache, rows, c, view.meta["normalization"])
    assert single.labels.sum() == len(rows)  # one event hits detector 11 in every file
    assert view.provenance("train", "hf") != view.provenance("train", "lf")
