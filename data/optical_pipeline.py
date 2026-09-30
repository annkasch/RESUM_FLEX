"""Prepare and reload fixed optical data without changing the model contract."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.batch_source import FixedBatchSource
from data.optical_features import AffineTransform, make_features, make_labels
from data.optical_reader import read_optical_runs
from data.optical_reports import generate_reports
from data.optical_split import FIDELITIES, SPLITS, assign_splits, save_manifest
from schemas.data_models import InputMode, StandardBatch
from schemas.optical import OpticalDataConfig


@dataclass
class PreparedOpticalData:
    batches: dict[str, dict[str, StandardBatch | list[StandardBatch]]]
    metadata: dict
    normalization: dict

    def training_source(self, fidelity="lf"):
        from data.grouped_batches import batch_groups, GroupedFixedBatchSource

        groups = batch_groups(self.batches["train"][fidelity])
        if len(groups) > 1:
            return GroupedFixedBatchSource(groups)
        return FixedBatchSource(groups[0], replace_events=False)


def load_prepared_batch(path: str | Path) -> StandardBatch:
    with np.load(path, allow_pickle=False) as arrays:
        return StandardBatch(
            mode=InputMode(str(arrays["mode"].item())),
            theta=arrays["theta"],
            labels=arrays["labels"],
            phi=arrays["phi"] if "phi" in arrays else None,
        )


def prepared_partition_paths(root, split, fidelity):
    folder = Path(root) / "batches" / split
    dense = folder / f"{fidelity}.npz"
    grouped = sorted(folder.glob(f"{fidelity}_*.npz"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if dense.exists() and grouped:
        raise ValueError(f"Ambiguous dense/grouped batches in {folder}")
    paths = [dense] if dense.exists() else grouped
    if not paths:
        raise FileNotFoundError(f"No prepared {split}/{fidelity} batches in {root}")
    return paths


def load_prepared_partition(root, split, fidelity):
    batches = [load_prepared_batch(p) for p in prepared_partition_paths(root, split, fidelity)]
    return batches[0] if len(batches) == 1 else batches


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
    training = [r for r in runs if assignments[r.file] == "train"]
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
    batches, metadata = {s: {} for s in SPLITS}, {}
    arrays_to_save = []
    for split in SPLITS:
        metadata[split] = {}
        for fidelity in FIDELITIES:
            selected = [r for r in runs if assignments[r.file] == split and r.fidelity == fidelity]
            if not selected:
                metadata[split][fidelity] = {"files": [], "status": "empty"}
                continue
            event_counts = sorted({r.n_events for r in selected})
            all_selected = selected
            partition_batches, partition_metadata = [], []
            for event_count in event_counts:
                selected = [r for r in all_selected if r.n_events == event_count]
                theta = theta_transform.transform(np.stack([features[r.file][0] for r in selected]))
                phi = (
                    phi_transform.transform(np.stack([features[r.file][1] for r in selected]))
                    if has_phi
                    else None
                )
                labels = np.stack([make_labels(r, config) for r in selected])
                mode = InputMode.FULL if has_phi else InputMode.DESIGN_ONLY
                batch = StandardBatch(mode=mode, theta=theta, phi=phi, labels=labels)
                partition_batches.append(batch)
                partition_metadata.append({
                    "files": [r.file for r in selected],
                    "nominal_centers": [r.nominal_center.tolist() for r in selected],
                    "measured_centers": [r.measured_center.tolist() for r in selected],
                    "features": features[selected[0].file][2],
                    "shape": list(labels.shape),
                    "mode": mode.value,
                })
                arrays = {
                    "theta": theta,
                    "labels": labels,
                    "mode": np.array(mode.value),
                    "event_ids": np.stack([r.event_ids for r in selected]),
                }
                if has_phi:
                    arrays["phi"] = phi
                suffix = fidelity if len(event_counts) == 1 else f"{fidelity}_{event_count}"
                arrays_to_save.append((split, suffix, arrays))
            batches[split][fidelity] = partition_batches[0] if len(partition_batches) == 1 else partition_batches
            metadata[split][fidelity] = partition_metadata[0] if len(partition_metadata) == 1 else {"groups": partition_metadata}
    out = config.output_directory
    # Validate everything before replacing prepared artifacts or the manifest.
    out.mkdir(parents=True, exist_ok=True)
    for split, fidelity, arrays in arrays_to_save:
        directory = out / "batches" / split
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"{fidelity}.npz", **arrays)
    # Remove obsolete dense/grouped files when a partition changes shape.
    expected = {out / "batches" / split / f"{name}.npz" for split, name, _ in arrays_to_save}
    for split in SPLITS:
        for fidelity in FIDELITIES:
            for p in (out / "batches" / split).glob(f"{fidelity}*.npz"):
                if p not in expected:
                    p.unlink()
    for name, payload in [
        ("metadata", metadata),
        ("normalization", normalization),
        ("config", config.model_dump(mode="json")),
    ]:
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    generate_reports(runs, assignments, excluded, config, manifest)
    save_manifest(config.split.manifest, manifest)
    (out / "manifest_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return PreparedOpticalData(batches, metadata, normalization)
