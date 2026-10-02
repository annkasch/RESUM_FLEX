import json

import numpy as np
import pytest

from core.experiments.storage import RunStore, snapshot_dataset, verify_dataset


def test_snapshot_is_deduplicated_and_detects_changes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    np.savez(source / "train.npz", x=[1, 2])
    snapshot = snapshot_dataset(source, tmp_path / "store")
    assert snapshot_dataset(source, tmp_path / "store") == snapshot
    np.savez(source / "train.npz", x=[3, 4])
    assert snapshot_dataset(source, tmp_path / "store") != snapshot
    verify_dataset(snapshot)
    (snapshot / "data/train.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="modified"):
        verify_dataset(snapshot)


def test_lifecycle_rebuild_and_immutable_artifacts(tmp_path):
    store = RunStore(tmp_path)
    path = store.create({"name": "baseline", "tags": ["test"]})
    store.transition(path, "running")
    (path / "metrics.json").write_text("[]")
    r = store.transition(path, "completed")
    assert store.inspect(r["id"])["status"] == "completed"
    with pytest.raises(ValueError, match="Immutable"):
        store.transition(path, "running")
    (tmp_path / "catalog.sqlite").unlink()
    assert store.rebuild()[0]["id"] == r["id"]
    assert store.list(tag="test")[0]["name"] == "baseline"
    (path / "metrics.json").write_text("[1]")
    with pytest.raises(ValueError, match="modified"):
        store.inspect(r["id"])
    failed = store.create({"name": "failed"})
    store.transition(failed, "failed", error="bad config")
    assert json.loads((failed / "run.json").read_text())["error"] == "bad config"
