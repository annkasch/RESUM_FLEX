"""Context-free event-model predictions for prepared design points, also without labels.

The mean predicted probability per design point is the input of the GP stage. Results are written
next to the checkpoint's run, never into the prepared data::

    <run>/inference/<checkpoint>/<partition>_<fidelity>.npz    per-design-point arrays
    <run>/inference/<checkpoint>/<partition>_<fidelity>.json   provenance and compute
    <run>/inference/<checkpoint>/<partition>_<fidelity>_events/ optional per-event logits

The per-event logits (``event_logits=True``, off by default) are float32, padding dropped,
in the provider's event order: design point by design point, each one's events in load order.
Shards hold whole design points (``shard_00000.npz``, ...); ``index.json`` lists each shard's
design point range.
``read_event_logits`` puts them back together.
"""

import json
from pathlib import Path

import numpy as np

from core.compute import ComputeLog
from core.surrogates.checkpoints import load_surrogate
from core.surrogates.datasets import load_provider
from core.surrogates.evaluation import _predict, evaluation_threads

# Marks "use the provider recorded in the checkpoint" (None means NpzBatches).
FROM_CHECKPOINT = "checkpoint"
# Events per event-logit shard (float32: 16 MB); a design point with more events gets its own shard.
SHARD_EVENTS = 2**22


def stream_units(model) -> bool:
    """True for a field trained on an event store (its own physical position units)."""
    module = getattr(model, "module", None)
    return bool(getattr(module, "uses_stream_units", False))


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
    events_from=None,
    seed=0,
    event_logits=False,
    shard_events=SHARD_EVENTS,
):
    """Predict every design point of a prepared partition with a context-free checkpoint.

    ``events_from`` = (partition, fidelity): each design point keeps its own theta but takes the
    event inputs (phi) of a randomly drawn donor design point of that partition (seeded). For event
    features defined relative to the design point (offset, direction), this is a fresh emission
    sample: the prediction then uses nothing from the design point's own simulation, as for a
    position that was never simulated. Default: the design point's own event inputs (no labels).

    ``data_provider`` defaults to the provider the checkpoint was trained with. The inputs
    must be normalized exactly as the training data: the provider's normalization must
    equal the one stored in the checkpoint. Returns the path of the written .npz file.

    ``event_logits`` also writes every event's logit (see the module docstring); with a
    Bayesian last layer these are the MAP logits. Not combinable with ``events_from``.
    """
    if event_logits and events_from is not None:
        raise ValueError("event_logits is not supported with events_from (donor event order)")
    compute = ComputeLog()
    checkpoint, data_directory = Path(checkpoint), Path(data_directory)
    model = load_surrogate(checkpoint)
    if model.uses_context:
        raise ValueError(
            f"{model.config.kind} needs labeled context events; unlabeled inference requires "
            "a context-free model (mlp, field, transformer, bdt)"
        )
    if data_provider == FROM_CHECKPOINT:
        data_provider = model.metadata.get("data_provider")
    provider = load_provider(data_provider, data_directory)
    if stream_units(model):
        # A field trained on ungrouped events works in physical units: map this data's
        # design point inputs (theta + offset) into them instead of requiring equal normalization.
        model.configure_normalization(provider.normalization())
    elif not normalization_matches(
        model.metadata.get("normalization.json"), provider.normalization()
    ):
        raise ValueError(
            f"Normalization of {data_directory} differs from the one {checkpoint} was trained "
            "with (or one of them does not record it); predictions would be invalid"
        )
    output = checkpoint.parent.parent / "inference" / checkpoint.name
    stem = output / f"{partition}_{fidelity}"
    if events_from is not None:
        stem = output / f"{partition}_{fidelity}_events-{events_from[0]}_{events_from[1]}"
    events_directory = stem.parent / f"{stem.name}_events"
    for existing in (stem.with_suffix(".npz"), events_directory):
        if existing.exists():
            raise FileExistsError(f"{existing} exists; inference results are never overwritten")
    with compute.stage("data_loading"):
        batch, data = provider.load(partition, fidelity, labels=False)
        ids = provider.theta(partition, fidelity)[1]
        donors = None
        if events_from is not None:
            donor = provider.load(*events_from, labels=False)[0]
            batch, donors = _donor_batch(batch, donor, seed)
    uncertainty = {}
    with evaluation_threads(threads), compute.stage("prediction"):
        prediction = _predict(model, batch, None)
        if getattr(model, "last_layer", None) is not None:
            from core.surrogates.last_layer import design_point_probability_draws

            draws = design_point_probability_draws(model, batch)  # [S, B]
            uncertainty = dict(
                probability_draws=draws.astype(np.float32),  # [S, B] last-layer draws
                probability_mean=draws.mean(axis=0),  # [B] posterior mean of the design point p
                probability_sd=draws.std(axis=0),  # [B] its epistemic sd
            )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        design_point_ids=np.array(ids),  # [B] provider design point identifiers
        coordinates=provider.coordinates(partition, fidelity),  # [B, D] physical
        theta=batch.theta,  # [B, D_theta] normalized model input
        event_counts=batch.events_per_design_point,  # [B]
        mean_probability=prediction.mean,  # [B] mean predicted p over each design point's events
        **uncertainty,
    )
    if event_logits:
        with compute.stage("event_logits"):
            write_event_logits(
                events_directory,
                prediction.logits,
                batch.events_per_design_point,
                np.array(ids),
                shard_events=shard_events,
            )
    provenance = dict(
        checkpoint=str(checkpoint.resolve()),
        step=model.metadata.get("step"),
        model=model.config.model_dump(mode="json"),
        data_provider=data_provider,
        data=data,
        partition=partition,
        fidelity=fidelity,
        design_points=int(batch.batch_size),
        events=int(batch.events_per_design_point.sum()),
        normalization_check=(
            "design point inputs mapped into the stream-trained field's physical units"
            if stream_units(model)
            else "provider normalization equals checkpoint normalization"
        ),
        labels="not used (inputs only)",
        uncertainty=(
            dict(
                method="Laplace last layer",
                draws=int(uncertainty["probability_draws"].shape[0]),
                seed=model.config.last_layer.seed,
            )
            if uncertainty
            else None
        ),
        event_inputs=(
            "own"
            if events_from is None
            else dict(
                donor_partition=events_from[0],
                donor_fidelity=events_from[1],
                seed=seed,
                donor_index=donors.tolist(),
            )
        ),
        event_logits=events_directory.name if event_logits else None,
        compute=compute.summary(),
    )
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2, allow_nan=False))
    return stem.with_suffix(".npz")


def write_event_logits(
    directory, logits, event_counts, design_point_ids, *, shard_events=SHARD_EVENTS
):
    """Write valid-event logits as float32 shards of whole design points plus ``index.json``.

    logits: [B, N] (positions beyond a design point's count are padding and dropped);
    event_counts: [B]; design_point_ids: [B]. ``directory`` must not exist.
    """
    logits, counts = np.asarray(logits), np.asarray(event_counts, dtype=np.int64)
    if (
        logits.ndim != 2
        or counts.shape != (logits.shape[0],)
        or len(design_point_ids) != len(counts)
    ):
        raise ValueError(
            f"Expected logits [B, N] with B event counts and design point ids; got {logits.shape}, "
            f"{counts.shape}, {len(design_point_ids)}"
        )
    if shard_events < 1:
        raise ValueError(f"shard_events must be positive, got {shard_events}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    shards, start = [], 0
    while start < len(counts):
        stop = start + 1
        total = counts[start]
        while stop < len(counts) and total + counts[stop] <= shard_events:
            total += counts[stop]
            stop += 1
        part = logits[start:stop]  # [b, N]
        valid = np.arange(part.shape[1]) < counts[start:stop, None]  # [b, N]
        name = f"shard_{len(shards):05d}.npz"
        np.savez(
            directory / name,
            logits=part[valid].astype(np.float32),  # [E] valid events, design point by design point
            event_counts=counts[start:stop],  # [b]
            design_point_ids=np.asarray(design_point_ids)[start:stop],  # [b]
        )
        shards.append(dict(file=name, design_points=[int(start), int(stop)], events=int(total)))
        start = stop
    index = dict(
        content=(
            "per-event logits of the design points' valid events, design point by design "
            "point in load order"
        ),
        dtype="float32",
        design_points=int(len(counts)),
        events=int(counts.sum()),
        shards=shards,
    )
    (directory / "index.json").write_text(json.dumps(index, indent=2))
    return directory


def read_event_logits(directory, design_points=None):
    """(design_point_ids [b], event_counts [b], logits [E]) from ``write_event_logits`` output.

    ``design_points`` = (start, stop) reads only that design point range, opening only the shards it
    touches; default: all design points.
    """
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    total = index["design_points"]
    start, stop = (0, total) if design_points is None else design_points
    if not 0 <= start <= stop <= total:
        raise ValueError(f"design point range {design_points} outside [0, {total}]")
    ids, counts, logits = [], [], []
    for shard in index["shards"]:
        first, last = shard["design_points"]
        if last <= start or first >= stop:
            continue
        with np.load(directory / shard["file"]) as arrays:
            shard_counts = arrays["event_counts"]
            offsets = np.concatenate([[0], np.cumsum(shard_counts)])
            lo, hi = max(start, first) - first, min(stop, last) - first
            ids.append(arrays["design_point_ids"][lo:hi])
            counts.append(shard_counts[lo:hi])
            logits.append(arrays["logits"][offsets[lo] : offsets[hi]])
    if not ids:
        return np.array([]), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    return np.concatenate(ids), np.concatenate(counts), np.concatenate(logits)


def _donor_batch(batch, donor, seed):
    """batch with each design point's phi replaced by a seeded random donor design point's phi."""
    from schemas.data_models import StandardBatch

    if donor.phi is None or batch.phi is None:
        raise ValueError("events_from needs event inputs (phi) in both partitions")
    index = np.random.default_rng(np.random.SeedSequence([seed, 0])).integers(
        donor.batch_size, size=batch.batch_size
    )
    counts = donor.events_per_design_point[index]
    return StandardBatch(
        mode=batch.mode,
        theta=batch.theta,
        phi=donor.phi[index],
        labels=np.zeros(donor.phi[index].shape[:2], dtype=np.int8),
        event_counts=counts,
    ), index
