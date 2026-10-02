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
    batches: dict[str, dict[str, StandardBatch]]
    metadata: dict
    normalization: dict

    def training_source(self, fidelity="lf"):
        return FixedBatchSource(self.batches["train"][fidelity], replace_events=False)


def load_prepared_batch(path: str | Path) -> StandardBatch:
    with np.load(path, allow_pickle=False) as arrays:
        return StandardBatch(
            mode=InputMode(str(arrays["mode"].item())),
            theta=arrays["theta"],
            labels=arrays["labels"],
            phi=arrays["phi"] if "phi" in arrays else None,
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
        if assignments[r.file] == "train"
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
    batches, metadata = {s: {} for s in SPLITS}, {}
    arrays_to_save = []
    for split in SPLITS:
        metadata[split] = {}
        for fidelity in FIDELITIES:
            selected = [r for r in runs if assignments[r.file] == split and r.fidelity == fidelity]
            if not selected:
                metadata[split][fidelity] = {"files": [], "status": "empty"}
                continue
            if len({r.n_events for r in selected}) != 1:
                raise ValueError(
                    f"{split}/{fidelity}: unequal event counts; no padding or truncation"
                )
            theta = theta_transform.transform(np.stack([features[r.file][0] for r in selected]))
            phi = (
                phi_transform.transform(np.stack([features[r.file][1] for r in selected]))
                if has_phi
                else None
            )
            labels = np.stack([make_labels(r, config) for r in selected])
            mode = InputMode.FULL if has_phi else InputMode.DESIGN_ONLY
            batch = StandardBatch(mode=mode, theta=theta, phi=phi, labels=labels)
            batches[split][fidelity] = batch
            metadata[split][fidelity] = {
                "files": [r.file for r in selected],
                "nominal_centers": [r.nominal_center.tolist() for r in selected],
                "measured_centers": [r.measured_center.tolist() for r in selected],
                "features": features[selected[0].file][2],
                "shape": list(labels.shape),
                "mode": mode.value,
            }
            arrays = {
                "theta": theta,
                "labels": labels,
                "mode": np.array(mode.value),
                "event_ids": np.stack([r.event_ids for r in selected]),
            }
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
    for split in SPLITS:
        for fidelity in FIDELITIES:
            p = out / "batches" / split / f"{fidelity}.npz"
            if fidelity not in batches[split] and p.exists():
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


def ensure_prepared_optical_data(config: OpticalDataConfig) -> Path:
    """Prepare absent/stale artifacts; reuse data only after checking raw inputs.

    Source/configuration changes automatically update the manifest and rebuild
    arrays and normalization. Compatible existing split assignments are retained;
    explicit split policies take precedence. Invalid inputs still fail validation.
    """
    out = config.output_directory
    runs, _ = read_optical_runs(config.source)
    _, expected_manifest = assign_splits(runs, config, update=True)
    required = [out / f"{name}.json" for name in ("config", "metadata", "normalization", "manifest_snapshot")]
    required += [out / "batches" / split / f"{fid}.npz"
                 for split, fid in sorted({(r["split"], r["fidelity"]) for r in expected_manifest["files"]})]
    valid = all(p.is_file() for p in required) and config.split.manifest.is_file()
    if valid:
        try:
            saved = OpticalDataConfig.model_validate(json.loads((out / "config.json").read_text()))
            valid = saved == config and json.loads((out / "manifest_snapshot.json").read_text()) == expected_manifest
        except (ValueError, OSError):
            valid = False
    if not valid:
        prepare_optical_data(config, update_manifest=True)
    return out
