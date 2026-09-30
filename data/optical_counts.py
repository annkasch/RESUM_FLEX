"""Aggregate whole optical runs to (position, hit-producing events, trials).

LF/HF tags control reporting and the split only, never GP inputs or likelihoods.
No event/context split, interpolation, oversampling or pseudo-counts.
"""

import json

import numpy as np

from data.optical_features import center, make_labels
from data.optical_reader import read_optical_runs
from data.optical_split import assign_splits, save_manifest


def prepare_optical_counts(config):
    if config.theta.coordinates != "cartesian":
        raise ValueError("Optical count-GP workflow currently uses Cartesian voxel centers")
    runs, excluded = read_optical_runs(config.source)
    assignments, manifest = assign_splits(runs, config, update=True)
    rows = []
    for run in runs:
        if not np.all(run.n_part == 1):
            raise ValueError(f"{run.file}: count-GP requires one primary per event")
        rows.append(
            dict(
                file=run.file,
                source_group=run.fidelity,
                split=assignments[run.file],
                theta=center(run, config).tolist(),
                hits=int(make_labels(run, config).sum()),
                trials=run.n_events,
                fingerprint=run.fingerprint,
            )
        )
    out = config.output_directory
    out.mkdir(parents=True, exist_ok=True)
    partitions = {}
    for split in ("train", "validation", "test"):
        selected = [r for r in rows if r["split"] == split]
        if not selected:
            continue
        arrays = dict(
            theta=np.asarray([r["theta"] for r in selected]),
            hits=np.array([r["hits"] for r in selected], dtype=np.int64),
            trials=np.array([r["trials"] for r in selected], dtype=np.int64),
            source_group=np.array([r["source_group"] for r in selected]),
            files=np.array([r["file"] for r in selected]),
        )
        partitions[split] = arrays
        np.savez_compressed(out / f"{split}.npz", **arrays)
    for split in ("train", "validation", "test"):
        if split not in partitions and (out / f"{split}.npz").exists():
            (out / f"{split}.npz").unlink()
    save_manifest(config.split.manifest, manifest)
    (out / "counts.json").write_text(json.dumps(rows, indent=2))
    (out / "config.json").write_text(config.model_dump_json(indent=2))
    return partitions, rows, manifest
