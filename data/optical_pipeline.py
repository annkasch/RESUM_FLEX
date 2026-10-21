"""Prepare and reload fixed optical data without changing the model contract."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.batch_source import FixedBatchSource
from data.optical_features import AffineTransform, make_features, make_labels
from data.optical_reader import read_optical_runs, source_stamp
from data.optical_reports import generate_reports
from data.optical_split import (
    FIDELITIES,
    assign_splits,
    partition_names,
    save_manifest,
    training_partition,
)
from schemas.data_models import InputMode, StandardBatch
from schemas.optical import OpticalDataConfig


@dataclass
class PreparedOpticalData:
    batches: dict[str, dict[str, StandardBatch]]
    metadata: dict
    normalization: dict

    def training_source(self, fidelity="lf"):
        split = "cnp_train" if "cnp_train" in self.batches else "train"
        return FixedBatchSource(self.batches[split][fidelity], replace_events=False)


def _pad(rows, width, fill=0):
    """Stack per-voxel event arrays, padding the event axis to ``width``."""
    out = np.full((len(rows), width, *rows[0].shape[1:]), fill, dtype=rows[0].dtype)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def load_prepared_batch(path: str | Path) -> StandardBatch:
    with np.load(path, allow_pickle=False) as arrays:
        return StandardBatch(
            mode=InputMode(str(arrays["mode"].item())),
            theta=arrays["theta"],
            labels=arrays["labels"],
            phi=arrays["phi"] if "phi" in arrays else None,
            event_counts=arrays["event_counts"] if "event_counts" in arrays else None,
        )


def prepare_optical_data(
    config: OpticalDataConfig, *, update_manifest=False
) -> PreparedOpticalData:
    runs, excluded = read_optical_runs(config.source)
    if config.target.kind == "single_channel" and not any(
        np.any(r.detector_ids == config.target.detector_id) for r in runs
    ):
        raise ValueError("Selected detector ID never observed; cannot validate it without a roster")
    assignments, manifest = assign_splits(runs, config, update=update_manifest)
    features = {r.file: make_features(r, config) for r in runs}
    training = [
        r for r in runs
        if assignments[r.file] == training_partition(config)
        and (config.normalization.fit_fidelities is None
             or r.fidelity in config.normalization.fit_fidelities)
    ]
    if not training:
        raise ValueError("No training files; adjust split or provide more data")
    theta_fit = np.stack([features[r.file][0] for r in training])
    theta_transform = AffineTransform.fit(theta_fit, config.normalization.method)
    has_phi = features[runs[0].file][1] is not None
    phi_transform = None
    if has_phi:
        phi_fit = np.concatenate([features[r.file][1] for r in training])
        phi_transform = AffineTransform.fit(phi_fit, config.normalization.method)
    normalization = {
        "theta": theta_transform.to_dict(),
        "phi": phi_transform.to_dict() if phi_transform else None,
        "fit_files": [r.file for r in training],
        "minmax_range": [0, 1],
    }
    splits = partition_names(config)
    batches, metadata = {s: {} for s in splits}, {}
    arrays_to_save = []
    for split in splits:
        metadata[split] = {}
        for fidelity in FIDELITIES:
            selected = [r for r in runs if assignments[r.file] == split and r.fidelity == fidelity]
            if not selected:
                metadata[split][fidelity] = {"files": [], "status": "empty"}
                continue
            # Unequal event counts are zero-padded to the largest voxel and
            # recorded in event_counts; equal counts keep the rectangular format.
            counts = np.array([r.n_events for r in selected], dtype=np.int64)
            width = int(counts.max())
            ragged = bool(np.any(counts != width))
            theta = theta_transform.transform(np.stack([features[r.file][0] for r in selected]))
            phi = (
                _pad([phi_transform.transform(features[r.file][1]) for r in selected], width)
                if has_phi
                else None
            )
            labels = _pad([make_labels(r, config) for r in selected], width)
            event_ids = _pad([r.event_ids for r in selected], width, fill=-1)
            mode = InputMode.FULL if has_phi else InputMode.DESIGN_ONLY
            batch = StandardBatch(
                mode=mode,
                theta=theta,
                phi=phi,
                labels=labels,
                event_counts=counts if ragged else None,
            )
            batches[split][fidelity] = batch
            metadata[split][fidelity] = {
                "files": [r.file for r in selected],
                "nominal_centers": [r.nominal_center.tolist() for r in selected],
                "measured_centers": [r.measured_center.tolist() for r in selected],
                "features": features[selected[0].file][2],
                "shape": list(labels.shape),
                "event_counts": counts.tolist() if ragged else None,
                "mode": mode.value,
            }
            arrays = {
                "theta": theta,
                "labels": labels,
                "mode": np.array(mode.value),
                "event_ids": event_ids,
            }
            if ragged:
                arrays["event_counts"] = counts
            if has_phi:
                arrays["phi"] = phi
            arrays_to_save.append((split, fidelity, arrays))
    out = config.output_directory
    # Validate everything before replacing prepared artifacts or the manifest.
    out.mkdir(parents=True, exist_ok=True)
    for split, fidelity, arrays in arrays_to_save:
        directory = out / "batches" / split
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"{fidelity}.npz", **arrays)
    # Remove stale batches when an updated manifest leaves a partition empty.
    for split in splits:
        for fidelity in FIDELITIES:
            p = out / "batches" / split / f"{fidelity}.npz"
            if fidelity not in batches[split] and p.exists():
                p.unlink()
    # The stamp lets ensure_prepared_optical_data skip re-reading unchanged raw files.
    for name, payload in [
        ("source_stamp", source_stamp(config.source)),
        ("metadata", metadata),
        ("normalization", normalization),
        ("config", config.model_dump(mode="json")),
    ]:
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    generate_reports(runs, assignments, excluded, config, manifest)
    save_manifest(config.split.manifest, manifest)
    (out / "manifest_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return PreparedOpticalData(batches, metadata, normalization)


def ensure_prepared_optical_data(config: OpticalDataConfig) -> Path:
    """Prepare absent/stale artifacts; reuse data only after checking raw inputs.

    Source/configuration changes automatically update the manifest and rebuild
    arrays and normalization. Compatible existing split assignments are retained;
    explicit split policies take precedence. Invalid inputs still fail validation.
    A matching configuration and raw-file stamp (name, size, mtime per file) reuses the data without
    reading raw files; any mismatch regenerates it.
    """
    out = config.output_directory
    stamp_path = out / "source_stamp.json"
    saved_stamp = json.loads(stamp_path.read_text()) if stamp_path.is_file() else None
    if (saved_stamp == source_stamp(config.source) and _matches_saved(config)
            and config.split.manifest.is_file() and _batches_complete(out)):
        return out
    # Something differs: regenerate directly, reading every raw file only once.
    prepare_optical_data(config, update_manifest=True)
    return out


def _matches_saved(config: OpticalDataConfig) -> bool:
    """True when the settings stored at the last preparation equal the current ones."""
    try:
        path = config.output_directory / "config.json"
        return OpticalDataConfig.model_validate(json.loads(path.read_text())) == config
    except (ValueError, OSError):
        return False


def _batches_complete(out: Path) -> bool:
    """Every partition in the saved manifest has its prepared batch file."""
    try:
        manifest = json.loads((out / "manifest_snapshot.json").read_text())
        metadata = json.loads((out / "metadata.json").read_text())
    except (ValueError, OSError):
        return False
    pairs = {(r["split"], r["fidelity"]) for r in manifest["files"]}
    return all((out / "batches" / s / f"{f}.npz").is_file() for s, f in pairs) and all(
        len(metadata.get(s, {}).get(f, {}).get("files", []))
        == sum((r["split"], r["fidelity"]) == (s, f) for r in manifest["files"])
        for s, f in pairs
    )
