"""Reuse optical features/splits while holding out a separate simulation budget."""

import csv
import json

import numpy as np
from scipy.spatial import cKDTree

from data.optical_features import AffineTransform, make_features, make_labels
from data.optical_pipeline import prepare_optical_data
from data.optical_reader import read_optical_runs
from schemas.data_models import InputMode, StandardBatch
from schemas.optical import OpticalDataConfig


def prepare_optical_transfer_data(config):
    """Keep all outcomes; exclude external tests within the spatial holdout radius.

    All normalization is fitted on the training split by prepare_optical_data.
    The test-source budget and geometry do not affect training assignments.
    """
    development, _ = read_optical_runs(config.source)
    external, excluded = read_optical_runs(config.test_source)
    for runs, budget in (
        (development, config.training_primaries),
        (external, config.test_primaries),
    ):
        if any(r.n_events != budget or np.any(r.n_part != 1) for r in runs):
            raise ValueError(f"Every included run must have {budget} single-primary events")
    # Test the geometry before writing prepared artifacts. Comparisons use physical
    # filename centers, not fitted normalization or observed labels.
    tree = cKDTree(np.stack([r.nominal_center for r in development]))
    distances, _ = tree.query(np.stack([r.nominal_center for r in external]))
    fingerprints = {r.fingerprint for r in development}
    retained = []
    for run, distance in zip(external, distances, strict=True):
        if run.fingerprint in fingerprints:
            raise ValueError("Test source reuses a training/validation simulation file")
        if distance <= config.spatial_separation_mm / 1000:
            excluded.append(
                {
                    "file": run.file,
                    "reason": "within spatial holdout radius",
                    "nearest_development_distance_m": float(distance),
                }
            )
        else:
            retained.append(run)
    if not retained:
        raise ValueError(
            "No spatially held-out test voxels remain; choose independent test locations"
        )
    base = OpticalDataConfig.model_validate(
        {name: getattr(config, name) for name in OpticalDataConfig.model_fields}
    )
    prepared = prepare_optical_data(base, update_manifest=True)
    if any("lf" not in prepared.batches[split] for split in ("train", "validation")):
        raise ValueError("Both training and validation require at least one LF1500 voxel group")
    transforms = prepared.normalization
    theta_transform = AffineTransform(**transforms["theta"])
    phi_transform = None if transforms["phi"] is None else AffineTransform(**transforms["phi"])
    features = [make_features(r, base) for r in retained]
    theta = theta_transform.transform(np.stack([f[0] for f in features]))
    phi = (
        None
        if phi_transform is None
        else phi_transform.transform(np.stack([f[1] for f in features]))
    )
    labels = np.stack([make_labels(r, base) for r in retained])
    batch = StandardBatch(
        mode=InputMode.DESIGN_ONLY if phi is None else InputMode.FULL,
        theta=theta,
        phi=phi,
        labels=labels,
    )
    prepared.batches["test"]["lf"] = batch
    prepared.metadata["test"]["lf"] = {
        "files": [r.file for r in retained],
        "nominal_centers": [r.nominal_center.tolist() for r in retained],
        "shape": list(labels.shape),
        "mode": batch.mode.value,
        "features": features[0][2],
        "primary_budget": config.test_primaries,
    }
    output = config.output_directory
    arrays = {
        "theta": theta,
        "labels": labels,
        "mode": np.array(batch.mode.value),
        "event_ids": np.stack([r.event_ids for r in retained]),
    }
    if phi is not None:
        arrays["phi"] = phi
    folder = output / "batches/test"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "lf.npz", **arrays)
    (output / "metadata.json").write_text(json.dumps(prepared.metadata, indent=2))
    (output / "transfer_config.json").write_text(config.model_dump_json(indent=2))
    (output / "test_source_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "file": r.file,
                        "fingerprint": r.fingerprint,
                        "nominal_center": r.nominal_center.tolist(),
                    }
                    for r in retained
                ],
                "excluded": excluded,
                "spatial_separation_mm": config.spatial_separation_mm,
                "independence": (
                    "Geometry and file identity checked; "
                    "simulator random-seed provenance not available"
                ),
            },
            indent=2,
        )
    )
    rows = []
    for split in ("train", "validation", "test"):
        b = prepared.batches[split]["lf"]
        counts = b.labels.sum(1)
        rows.append(
            {
                "split": split,
                "primaries_per_voxel": b.n_events,
                "voxels": b.batch_size,
                "zero_hit_voxels": int((counts == 0).sum()),
                "nonzero_hit_voxels": int((counts > 0).sum()),
                "events": int(b.labels.size),
                "positive_events": int(counts.sum()),
            }
        )
    with (output / "transfer_data_counts.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return prepared
