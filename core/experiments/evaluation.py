"""Versioned target manifests and metrics computed solely from prediction records."""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from core.experiments.storage import digest, file_hash, write_json
from core.gp_metrics import gp_metrics
from core.precision_recall import precision_recall
from schemas.data_models import valid_event_mask


def build_manifest(snapshot, settings, output):
    root, output = Path(snapshot) / "data", Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = (
        json.loads((root / "metadata.json").read_text())
        if (root / "metadata.json").exists()
        else {}
    )
    normpath = root / "normalization.json"
    norm = json.loads(normpath.read_text()) if normpath.exists() else {}
    manifest = {
        "schema_version": 1,
        "settings": settings.model_dump(mode="json"),
        "normalization_hash": file_hash(normpath) if normpath.exists() else None,
        "partitions": {},
    }
    for partition in settings.partitions:
        split, fid = partition.split("/")
        eventpath = root / "batches" / split / f"{fid}.npz"
        if eventpath.exists():
            with np.load(eventpath) as a:
                labels = a["labels"]
                nvox, nevents = labels.shape
                counts = (
                    a["event_counts"] if "event_counts" in a else np.full(nvox, nevents)
                ).astype(np.int64)
                if settings.context_events >= counts.min():
                    raise ValueError(f"{partition}: no target events left")
                rng = np.random.default_rng(settings.seed)
                # Unequal counts: permute each voxel's valid events; pad target slots.
                nc, width = settings.context_events, nevents - settings.context_events
                ctx = np.zeros((nvox, nc), dtype=np.int64)
                tgt = np.zeros((nvox, width), dtype=np.int64)
                for row, count in enumerate(counts):
                    permutation = rng.permutation(int(count))
                    ctx[row], tgt[row, : count - nc] = permutation[:nc], permutation[nc:]
                target_counts = counts - nc
                valid = np.arange(width)[None, :] < target_counts[:, None]
                ids = a["event_ids"] if "event_ids" in a else np.tile(np.arange(nevents), (nvox, 1))
                targets = np.where(valid, np.take_along_axis(labels, tgt, axis=1), 0)
                partition_meta = metadata.get(split, {}).get(fid, {})
                files = partition_meta.get("files") or [f"{partition}/row{i}" for i in range(nvox)]
                coordinates = partition_meta.get("nominal_centers")
                if coordinates is None and "theta" in a:
                    coordinates = a["theta"]
                    if norm.get("theta"):
                        coordinates = coordinates * np.asarray(norm["theta"]["scale"]) + np.asarray(
                            norm["theta"]["offset"]
                        )
                arrays = dict(
                    context_indices=ctx,
                    target_indices=tgt,
                    context_event_ids=np.take_along_axis(ids, ctx, axis=1),
                    target_event_ids=np.where(valid, np.take_along_axis(ids, tgt, axis=1), -1),
                    labels=targets,
                    hits=targets.sum(1).astype(np.int64),
                    trials=target_counts,
                    files=np.asarray(files),
                )
                if not valid.all():
                    arrays["target_counts"] = target_counts
                if coordinates is not None:
                    arrays["coordinates"] = np.asarray(coordinates)
                id_quality = "source_event_ids" if "event_ids" in a else "prepared_row_indices"
        else:
            countpath = root / f"{split}.npz"
            if not countpath.exists():
                raise FileNotFoundError(f"Missing evaluation partition {partition}")
            if settings.context_events:
                raise ValueError("Cannot remove context events from aggregate counts")
            with np.load(countpath) as a:
                chosen = a["source_group"] == fid
                if not chosen.any():
                    raise ValueError(f"Empty evaluation partition {partition}")
                arrays = {k: a[k][chosen] for k in ("hits", "trials", "files")}
                arrays["coordinates"] = a["theta"][chosen]
            id_quality = "aggregate_only"
        nvox = len(arrays["hits"])
        arrays["weights"] = np.full(nvox, 1 / nvox)
        filename = partition.replace("/", "_") + ".npz"
        np.savez_compressed(output / filename, **arrays)
        # JSON hashes are stable across NPZ compression and array dtype variations.
        target_contract = {
            k: v.tolist()
            for k, v in arrays.items()
            if k in {"files", "target_event_ids", "hits", "trials", "coordinates", "weights"}
        }
        target_contract["event_identity"] = id_quality
        manifest["partitions"][partition] = {
            "file": filename,
            "sha256": file_hash(output / filename),
            "target_id": digest(target_contract),
            "event_identity": id_quality,
            "voxels": nvox,
            "target_events": int(arrays["trials"].sum()),
            "context_events_per_voxel": settings.context_events,
            "weighting": settings.weighting,
        }
    manifest["id"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def save_record(
    output,
    name,
    arrays,
    *,
    partition,
    protocol,
    model,
    checkpoint,
    quantity,
    distribution=None,
    conditioning=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entry = protocol["partitions"][partition]
    a = {k: np.asarray(v) for k, v in arrays.items()}
    if not np.isfinite(a["mean"]).all():
        raise ValueError("Nonfinite prediction")
    np.savez_compressed(output / f"{name}.npz", **a)
    record = dict(
        schema_version=1,
        name=name,
        partition=partition,
        model=model,
        checkpoint=checkpoint,
        target_id=entry["target_id"],
        evaluation_id=protocol["id"],
        quantity=quantity,
        distribution=distribution,
        conditioning=conditioning or {},
        file=f"{name}.npz",
        sha256=file_hash(output / f"{name}.npz"),
    )
    write_json(output / f"{name}.json", record)
    return record


def score_record(directory, record):
    path = Path(directory) / record["file"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("Saved prediction record was modified")
    with np.load(path) as a:
        intervals = {k: (a[f"lower_{k}"], a[f"upper_{k}"]) for k in (1, 2, 3) if f"lower_{k}" in a}
        result = gp_metrics(
            a["observed"],
            a["mean"],
            intervals=intervals,
            samples=a["samples"] if "samples" in a else None,
            median=a["median"] if "median" in a else None,
        )
        if "logits" in a:
            labels, logits = a["labels"], a["logits"]
            if labels.shape != logits.shape or not np.isin(labels, [0, 1]).all():
                raise ValueError("Event metrics require aligned real binary targets")
            if "event_counts" in a:
                # Drop padding of unequal per-voxel target counts.
                mask = valid_event_mask(a["event_counts"], labels.shape)
                labels, logits = labels[mask], logits[mask]
            curve = precision_recall(labels.ravel(), logits.ravel()) if labels.sum() else None
            result.update(
                bernoulli_log_loss=float(
                    np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits)).mean()
                ),
                brier_score=float(np.square(expit(logits) - labels).mean()),
                average_precision=None if curve is None else curve["average_precision"],
                prevalence=float(labels.mean()),
            )
        if "log_probability" in a:
            result["predictive_nll"] = (
                float(-a["log_probability"].mean())
                if np.isfinite(a["log_probability"]).all()
                else None
            )
            result["nll_measure"] = record["distribution"]["measure"]
        result["outside_probability_range"] = int(((a["mean"] < 0) | (a["mean"] > 1)).sum())
    return {**record, "metrics": result}


def evaluate_records(directory):
    directory = Path(directory)
    return [
        score_record(directory, json.loads(p.read_text())) for p in sorted(directory.glob("*.json"))
    ]
