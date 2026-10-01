"""Saved MFGP projection workflow; numerical engine is backend independent."""

import hashlib
import json
from pathlib import Path

import numpy as np

from core.projection_adapters import BinomialObservation, MFGPProjectionAdapter
from core.spatial_projections import (
    PROJECTIONS as PROJECTIONS,
)
from core.spatial_projections import (
    average_grid as average_grid,
)
from core.spatial_projections import (
    project_response,
)
from core.spatial_projections import (
    projection_grid as projection_grid,
)
from core.spatial_projections import (
    sample_locations as sample_locations,
)


def marginalize_posterior(gp, training_theta, offset, scale, settings):
    """Compatibility entry point using the common engine and MFGP adapter."""
    observation = None
    if settings.quantity == "observed_fraction":
        if settings.target_events is None:
            raise ValueError("Observed-fraction projections require target_events")
        observation = BinomialObservation(settings.target_events)
    return project_response(
        MFGPProjectionAdapter(gp, offset=offset, scale=scale),
        training_theta,
        settings,
        observation_model=observation,
    )


def ensure_mfgp_projections(directory, settings):
    """Create cached projections from a saved run; never retrain or reread raw data."""
    from core.surrogate_mfgp import load_mfgp
    from viz.mfgp_projections import plot_mfgp_projections

    directory = Path(directory)
    sources = [
        directory / name for name in ("model.pkl", "training_arrays.npz", "hf_validation.npz")
    ]
    if settings.quantity == "observed_fraction" and settings.target_events is None:
        with np.load(directory / "hf_validation.npz") as validation:
            n_target = int(validation["target_events"]) if "target_events" in validation else None
        if n_target is None:
            # Older runs: recover N only from the exact prepared batch used in that run.
            model_meta = json.loads((directory / "model.json").read_text())
            experiment = json.loads((directory.parent / "experiment.json").read_text())
            record = model_meta["data"]["validation/hf"]
            batch_path = Path(record["path"])
            if (
                not batch_path.exists()
                or hashlib.sha256(batch_path.read_bytes()).hexdigest() != record["sha256"]
            ):
                raise ValueError(
                    "Cannot recover saved target N; set projections.target_events explicitly"
                )
            with np.load(batch_path) as batch:
                n_target = (
                    batch["labels"].shape[1]
                    - experiment["resolved_config"]["validation_context_events"]
                )
        if n_target <= 0:
            raise ValueError("Saved target-event count must be positive")
        settings = settings.model_copy(update={"target_events": n_target})
    norm_path = directory / "normalization.json"
    if norm_path.exists():
        sources.append(norm_path)
    signature = dict(
        version=4,
        config=settings.model_dump(mode="json"),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    output = directory / (
        "projections_observed_fraction"
        if settings.quantity == "observed_fraction"
        else "projections"
    )
    metadata_path = output / "metadata.json"
    data_path = output / "projections.npz"
    valid = False
    if metadata_path.exists() and data_path.exists():
        valid = json.loads(metadata_path.read_text()).get("signature") == signature
    if not valid:
        offset, scale = np.zeros(3), np.ones(3)
        if norm_path.exists():
            norm = json.loads(norm_path.read_text())["theta"]
            offset, scale = np.asarray(norm["offset"]), np.asarray(norm["scale"])
        if offset.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0):
            raise ValueError("Projections require a valid three-coordinate affine normalization")
        with np.load(directory / "training_arrays.npz") as data:
            training_theta = np.concatenate((data["X_lf"], data["X_hf"])) * scale + offset
            observed_train_theta = data["X_hf"] * scale + offset
            observed_train = data["Y_hf_raw"].ravel()
        gp = load_mfgp(directory / "model.pkl")
        arrays, metadata = marginalize_posterior(gp, training_theta, offset, scale, settings)
        arrays.update(observed_train_theta=observed_train_theta, observed_train=observed_train)
        with np.load(directory / "hf_validation.npz") as data:
            arrays.update(
                observed_validation_theta=data["theta"] * scale + offset,
                observed_validation=data["observed"],
            )
        output.mkdir(exist_ok=True)
        np.savez_compressed(data_path, **arrays)
        metadata_path.write_text(json.dumps(dict(signature=signature, **metadata), indent=2))
    names = ("curves", "curves_observed", "planes", "planes_observed")
    missing_counts = settings.quantity == "observed_fraction" and (
        not (output / "coverage_counts.json").exists()
        or json.loads((output / "coverage_counts.json").read_text()).get("plot_version") != 2
    )
    if (
        not valid
        or missing_counts
        or any(not (output / f"{name}.{ext}").exists() for name in names for ext in ("png", "pdf"))
    ):
        plot_mfgp_projections(output)
    return output
