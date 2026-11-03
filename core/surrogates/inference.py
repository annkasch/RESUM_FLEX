"""Context-free event-model predictions for prepared voxels, also without labels.

The mean predicted probability per voxel is the input of the GP stage. Results are written
next to the checkpoint's run, never into the prepared data::

    <run>/inference/<checkpoint>/<partition>_<fidelity>.npz    per-voxel arrays
    <run>/inference/<checkpoint>/<partition>_<fidelity>.json   provenance
"""

import json
from pathlib import Path

import numpy as np

from core.surrogates.checkpoints import load_surrogate
from core.surrogates.datasets import load_provider
from core.surrogates.evaluation import _predict, evaluation_threads

# Marks "use the provider recorded in the checkpoint" (None means NpzBatches).
FROM_CHECKPOINT = "checkpoint"


def normalization_matches(saved: dict | None, current: dict | None) -> bool:
    """True when both normalizations exist and agree on theta and phi."""
    if saved is None or current is None:
        return False
    return all(saved.get(key) == current.get(key) for key in ("theta", "phi"))


def predict_partition(
    checkpoint,
    data_directory,
    partition,
    fidelity="lf",
    *,
    threads=1,
    data_provider=FROM_CHECKPOINT,
    photons_from=None,
    seed=0,
):
    """Predict every voxel of a prepared partition with a context-free checkpoint.

    ``photons_from`` = (partition, fidelity): each voxel keeps its own theta but takes the
    event inputs (phi) of a randomly drawn donor voxel of that partition (seeded). For event
    features defined relative to the voxel (offset, direction), this is a fresh emission
    sample: the prediction then uses nothing from the voxel's own simulation, as for a
    position that was never simulated. Default: the voxel's own event inputs (no labels).

    ``data_provider`` defaults to the provider the checkpoint was trained with. The inputs
    must be normalized exactly as the training data: the provider's normalization must
    equal the one stored in the checkpoint. Returns the path of the written .npz file.
    """
    checkpoint, data_directory = Path(checkpoint), Path(data_directory)
    model = load_surrogate(checkpoint)
    if model.uses_context:
        raise ValueError(
            f"{model.config.kind} needs labeled context events; unlabeled inference requires "
            "a context-free model (mlp, transformer, bdt)"
        )
    if data_provider == FROM_CHECKPOINT:
        data_provider = model.metadata.get("data_provider")
    provider = load_provider(data_provider, data_directory)
    if not normalization_matches(
        model.metadata.get("normalization.json"), provider.normalization()
    ):
        raise ValueError(
            f"Normalization of {data_directory} differs from the one {checkpoint} was trained "
            "with (or one of them does not record it); predictions would be invalid"
        )
    output = checkpoint.parent.parent / "inference" / checkpoint.name
    stem = output / f"{partition}_{fidelity}"
    if photons_from is not None:
        stem = output / f"{partition}_{fidelity}_photons-{photons_from[0]}_{photons_from[1]}"
    if stem.with_suffix(".npz").exists():
        raise FileExistsError(f"{stem}.npz exists; inference results are never overwritten")
    batch, data = provider.load(partition, fidelity, labels=False)
    ids = provider.theta(partition, fidelity)[1]
    donors = None
    if photons_from is not None:
        batch, donors = _donor_batch(batch, provider.load(*photons_from, labels=False)[0], seed)
    with evaluation_threads(threads):
        prediction = _predict(model, batch, None)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        voxel_ids=np.array(ids),  # [B] provider voxel identifiers
        coordinates=provider.coordinates(partition, fidelity),  # [B, D] physical
        theta=batch.theta,  # [B, D_theta] normalized model input
        event_counts=batch.events_per_voxel,  # [B]
        mean_probability=prediction.mean,  # [B] mean predicted p over each voxel's events
    )
    provenance = dict(
        checkpoint=str(checkpoint.resolve()),
        step=model.metadata.get("step"),
        model=model.config.model_dump(mode="json"),
        data_provider=data_provider,
        data=data,
        partition=partition,
        fidelity=fidelity,
        voxels=int(batch.batch_size),
        events=int(batch.events_per_voxel.sum()),
        normalization_check="provider normalization equals checkpoint normalization",
        labels="not used (inputs only)",
        event_inputs=(
            "own" if photons_from is None
            else dict(donor_partition=photons_from[0], donor_fidelity=photons_from[1],
                      seed=seed, donor_index=donors.tolist())
        ),
    )
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2, allow_nan=False))
    return stem.with_suffix(".npz")


def _donor_batch(batch, donor, seed):
    """batch with each voxel's phi replaced by a seeded random donor voxel's phi."""
    from schemas.data_models import StandardBatch

    if donor.phi is None or batch.phi is None:
        raise ValueError("photons_from needs event inputs (phi) in both partitions")
    index = np.random.default_rng(np.random.SeedSequence([seed, 0])).integers(
        donor.batch_size, size=batch.batch_size
    )
    counts = donor.events_per_voxel[index]
    return StandardBatch(
        mode=batch.mode,
        theta=batch.theta,
        phi=donor.phi[index],
        labels=np.zeros(donor.phi[index].shape[:2], dtype=np.int8),
        event_counts=counts,
    ), index
