"""Context-free event-model predictions for prepared voxels without optical labels.

The mean predicted probability per voxel is the input of the GP stage. Results are written
next to the checkpoint's run, never into the raw data or the event cache::

    <run>/inference/<checkpoint>/<partition>_<fidelity>.npz    per-voxel arrays
    <run>/inference/<checkpoint>/<partition>_<fidelity>.json   provenance
"""

import json
from pathlib import Path

import numpy as np

from core.surrogates.checkpoints import load_surrogate
from core.surrogates.evaluation import _predict, evaluation_threads


def normalization_matches(saved: dict | None, current: dict | None) -> bool:
    """True when both normalizations exist and agree on theta and phi."""
    if saved is None or current is None:
        return False
    return all(saved.get(key) == current.get(key) for key in ("theta", "phi"))


def predict_partition(checkpoint, data_directory, partition, fidelity="lf", *, threads=1):
    """Predict every voxel of a prepared partition with a context-free checkpoint.

    The partition's inputs must be normalized exactly as the model's training data:
    the view normalization must equal the one stored in the checkpoint metadata.
    Returns the path of the written .npz file.
    """
    from data.optical_view import OpticalView

    checkpoint, data_directory = Path(checkpoint), Path(data_directory)
    model = load_surrogate(checkpoint)
    if model.uses_context:
        raise ValueError(
            f"{model.config.kind} needs labeled context events; unlabeled inference requires "
            "a context-free model (mlp, transformer, bdt)"
        )
    view = OpticalView(data_directory)
    if not normalization_matches(
        model.metadata.get("normalization.json"), view.meta["normalization"]
    ):
        raise ValueError(
            f"Normalization of {data_directory} differs from the one {checkpoint} was trained "
            "with (or the checkpoint does not record it); predictions would be invalid"
        )
    output = checkpoint.parent.parent / "inference" / checkpoint.name
    stem = output / f"{partition}_{fidelity}"
    if stem.with_suffix(".npz").exists():
        raise FileExistsError(f"{stem}.npz exists; inference results are never overwritten")
    batch, paths = view.load_inputs(partition, fidelity)
    rows = view.cache.locate(paths)
    with evaluation_threads(threads):
        prediction = _predict(model, batch, None)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        files=np.array(paths),
        nominal_center=view.cache.index["nominal_center"][rows],  # [B, 3] m
        theta=batch.theta,  # [B, D_theta] normalized model input
        event_counts=batch.events_per_voxel,  # [B]
        mean_probability=prediction.mean,  # [B] mean predicted p over each voxel's events
    )
    provenance = dict(
        checkpoint=str(checkpoint.resolve()),
        step=model.metadata.get("step"),
        model=model.config.model_dump(mode="json"),
        data=view.provenance(partition, fidelity),
        partition=partition,
        fidelity=fidelity,
        voxels=int(batch.batch_size),
        events=int(batch.events_per_voxel.sum()),
        normalization_check="view normalization equals checkpoint normalization",
        labels="not used (inputs only)",
    )
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2, allow_nan=False))
    return stem.with_suffix(".npz")
