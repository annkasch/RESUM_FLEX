"""Reuse optical features/splits while holding out a separate simulation budget."""

import csv
import json

import numpy as np
from scipy.spatial import cKDTree

from data.event_cache import EventCache, cached_runs
from data.optical_pipeline import prepare_optical_data
from schemas.optical import OpticalDataConfig


def prepare_optical_transfer_data(config):
    """Keep all outcomes; exclude external tests within the spatial holdout radius.

    All normalization is fitted on the training split by prepare_optical_data.
    The test-source budget and geometry do not affect training assignments.
    """
    cache = EventCache(config.event_cache)
    development, _ = cached_runs(config.source, cache)
    external, excluded = cached_runs(config.test_source, cache)
    for runs, budget in (
        ([r for r in development if r.fidelity == "lf"], config.training_primaries),
        ([r for r in development if r.fidelity == "hf"], config.hf_primaries),
        (external, config.test_primaries),
    ):
        if any(r.n_events != budget or r.max_n_part != 1 for r in runs):
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
    # Normalization is fitted on the training split only; test files just join the view.
    prepared = prepare_optical_data(
        base, update_manifest=True, extra=[("test", r) for r in retained]
    )
    if any("lf" not in prepared.available(split) for split in ("train", "validation")):
        raise ValueError("Both training and validation require at least one LF1500 voxel group")
    prepared.metadata["test"]["lf"]["primary_budget"] = config.test_primaries
    output = config.output_directory
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
        for fidelity in prepared.available(split):
            b = prepared.batch(split, fidelity)
            counts = b.labels.sum(1)
            rows.append(
                {
                    "split": split,
                    "fidelity": fidelity,
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
