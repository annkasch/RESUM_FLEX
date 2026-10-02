import numpy as np
import pytest

from core.experiments.evaluation import build_manifest, save_record, score_record
from core.experiments.storage import snapshot_dataset
from schemas.experiments import EvaluationSpec


def make_events(path):
    folder = path / "batches/validation"
    folder.mkdir(parents=True)
    np.savez(
        folder / "hf.npz",
        labels=[[0, 1, 0, 0], [0, 0, 1, 0]],
        theta=[[0.0, 1, 2], [2.0, 3, 4]],
        event_ids=[[10, 11, 12, 13], [20, 21, 22, 23]],
    )


def test_target_manifest_and_prediction_roundtrip(tmp_path):
    make_events(tmp_path / "data")
    snapshot = snapshot_dataset(tmp_path / "data", tmp_path / "store")
    settings = EvaluationSpec(context_events=1, partitions=["validation/hf"])
    a = build_manifest(snapshot, settings, tmp_path / "a")
    b = build_manifest(snapshot, settings, tmp_path / "b")
    assert a == b
    c = build_manifest(snapshot, settings.model_copy(update={"seed": 10}), tmp_path / "c")
    assert (
        a["partitions"]["validation/hf"]["target_id"]
        != c["partitions"]["validation/hf"]["target_id"]
    )
    with np.load(tmp_path / "a/validation_hf.npz") as ids:
        assert not set(ids["context_event_ids"][0]) & set(ids["target_event_ids"][0])
        arrays = dict(
            observed=ids["hits"] / ids["trials"],
            mean=np.array([0.2, 0.3]),
            logits=np.zeros((2, 3)),
            labels=ids["labels"],
        )
    r = save_record(
        tmp_path / "records",
        "test",
        arrays,
        partition="validation/hf",
        protocol=a,
        model="mlp",
        checkpoint="saved",
        quantity="event_score_mean",
    )
    metrics = score_record(tmp_path / "records", r)["metrics"]
    assert metrics["bernoulli_log_loss"] == pytest.approx(np.log(2))
    assert metrics["mean_bias"] == pytest.approx(0.25 - arrays["observed"].mean())
    assert score_record(tmp_path / "records", r)["metrics"] == metrics
    (tmp_path / "records/test.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        score_record(tmp_path / "records", r)


def test_aggregate_counts_do_not_claim_event_identity(tmp_path):
    root = tmp_path / "counts"
    root.mkdir()
    np.savez(
        root / "validation.npz",
        theta=[[0.0, 1.0, 2.0]],
        hits=[1],
        trials=[100],
        files=["run1"],
        source_group=["hf"],
    )
    snapshot = snapshot_dataset(root, tmp_path / "store")
    settings = EvaluationSpec(context_events=0, partitions=["validation/hf"])
    manifest = build_manifest(snapshot, settings, tmp_path / "manifest")
    assert manifest["partitions"]["validation/hf"]["event_identity"] == "aggregate_only"
    with pytest.raises(ValueError, match="remove context"):
        build_manifest(
            snapshot, settings.model_copy(update={"context_events": 1}), tmp_path / "bad"
        )
