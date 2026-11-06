"""Train a neural field on every event of an event store (``python -m core.surrogates stream``).

Each optimizer step takes one batch of raw event records from the event source (random
order, see its documentation); the field normalizes positions with the store's train
normalization (``BernoulliField.set_stream_normalization``) and encodes the direction from the
momentum. Checkpoints are selected by the Bernoulli log loss on the first
``validation_events`` validation events, held on the device. With ``last_layer`` the
selected checkpoints get a Laplace posterior from one streaming pass over all training
events (exact Bernoulli Hessian of the final layer, as for voxel training).

Several GPUs: launch with ``torchrun``; every rank reads its own blocks of each pass and
DistributedDataParallel averages the gradients. Rank 0 validates and writes the outputs.

The checkpoints are ordinary field checkpoints: ``predict`` / ``score_event_model.py``
evaluate them on voxel data, mapping theta + offset into the field's stream units.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

from core.compute import ComputeLog
from core.event_stream import event_labels, open_event_source
from core.surrogates.base import Episode
from core.surrogates.checkpoints import load_surrogate, save_surrogate
from core.surrogates.evaluation import calibration_bins, evaluate_surrogate
from core.surrogates.last_layer import _layer, last_layer_weights
from core.surrogates.models import build_surrogate
from schemas.surrogates import StreamRunConfig

VALIDATION_CHUNK = 2**20


def _distributed():
    """(rank, world size, local rank) from torchrun's environment; (0, 1, 0) otherwise."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, 0
    return int(os.environ["RANK"]), world, int(os.environ.get("LOCAL_RANK", "0"))


def _stream_module(field):
    import torch

    class StreamForward(torch.nn.Module):
        """forward(records) = field.forward_events(records), for DDP and torch.compile."""

        def __init__(self, field):
            super().__init__()
            self.field = field

        def forward(self, records):
            return self.field.forward_events(records)

    return StreamForward(field)


def _log_loss(forward, events, device, autocast) -> float:
    """Mean Bernoulli log loss of event records (on ``device``), in chunks, no gradients."""
    import torch

    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(events), VALIDATION_CHUNK):
            chunk = events[start : start + VALIDATION_CHUNK]
            with autocast():
                logits = forward(chunk)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), event_labels(chunk), reduction="sum"
            )
            total += float(loss)
            count += len(chunk)
    return total / count


def fit_stream_last_layer(model, source, *, batch_size, device):
    """Laplace posterior of the final layer from one pass over all events of ``source``."""
    import torch

    from core.last_layer import laplace_posterior

    settings = model.config.last_layer
    layer = _layer(model)
    mean = last_layer_weights(model)
    captured = []
    handle = layer.register_forward_hook(lambda _m, inputs, _o: captured.append(inputs[0]))
    field = model.module
    training = field.training
    hessian = torch.zeros((len(mean), len(mean)), dtype=torch.float64, device=device)
    weights = torch.as_tensor(mean, dtype=torch.float64, device=device)
    events = 0
    try:
        field.eval()
        with torch.no_grad():
            for batch in source.batches(batch_size, 0, device=device):
                captured.clear()
                field.forward_events(batch)
                features = captured[-1].double()
                features = torch.cat([features, features.new_ones((len(features), 1))], dim=1)
                p = torch.sigmoid(features @ weights)
                hessian += (features * (p * (1 - p))[:, None]).T @ features
                events += len(features)
    finally:
        handle.remove()
        field.train(training)
    posterior = laplace_posterior(
        hessian.cpu().numpy(), mean, prior_precision=settings.prior_precision, events=events
    )
    model.last_layer = posterior
    model.metadata["last_layer"] = dict(
        prior_precision=posterior.prior_precision,
        prior_precision_source=("evidence" if settings.prior_precision == "evidence" else "fixed"),
        log_evidence=posterior.log_evidence,
        events=events,
        parameters=len(mean),
        hessian="exact Bernoulli Hessian of the final layer over all training events",
    )
    return posterior


def _voxel_episodes(config: StreamRunConfig):
    """Labelled voxel episodes of ``config.evaluation`` (context-free: no context events).

    Returns (episodes {(split, fidelity): Episode}, the history episode, data record).
    """
    from core.gp_subsample import subsample_voxels
    from core.surrogates.datasets import load_provider, prepare_data

    if config.data_config is not None:
        prepare_data(config.data_provider, config.data_config, config.data_directory)
    provider = load_provider(config.data_provider, config.data_directory)
    episodes, provenance = {}, {}
    for item in config.evaluation:
        batch, provenance[f"{item.split}/{item.fidelity}"] = provider.load(
            item.partition, item.fidelity
        )
        provenance[f"{item.split}/{item.fidelity}"]["partition"] = item.partition
        episodes[item.split, item.fidelity] = Episode(None, batch)
    first = next((e for e in config.evaluation if e.split == "validation"), None)
    history_episode, history_voxels = None, None
    if first is not None:
        target = episodes["validation", first.fidelity].target
        index = subsample_voxels(
            target.batch_size, config.history_voxels, config.evaluation_seed, 0
        )
        history_episode = Episode(
            None, target.take(index) if len(index) < target.batch_size else target
        )
        history_voxels = dict(
            set=f"validation/{first.fidelity}", available=target.batch_size, used=len(index)
        )
    data = dict(
        normalization=provider.normalization(), provenance=provenance, history_voxels=history_voxels
    )
    return episodes, history_episode, data


def stream_metrics(labels, logits, *, reference_rate):
    """Event metrics of events without voxel structure, as in voxel evaluation.

    labels, logits: [M] numpy. Returns (metrics, arrays) with calibration bins and the
    precision-recall curve.
    """
    from scipy.special import expit

    from core.precision_recall import precision_recall

    labels = np.asarray(labels, dtype=np.int8)
    logits = np.asarray(logits, dtype=np.float64)
    p = expit(logits)
    curve = precision_recall(labels, logits)
    log_loss = float(
        np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits)).mean()
    )
    reference = float(
        -np.where(labels == 1, np.log(reference_rate), np.log1p(-reference_rate)).mean()
    )
    bins = calibration_bins(labels, p)
    chi2 = (
        np.square(bins["calibration_hits"] - bins["calibration_predicted"])
        / bins["calibration_variance"]
    )
    variance = float((p * (1 - p)).sum())
    metrics = dict(
        events=int(len(labels)),
        positives=int(labels.sum()),
        mean_observed=float(labels.mean()),
        mean_predicted=float(p.mean()),
        bernoulli_log_loss=log_loss,
        reference_log_loss=reference,
        log_loss_skill=1 - log_loss / reference,
        brier_score=float(np.square(p - labels).mean()),
        average_precision=curve["average_precision"],
        prevalence=curve["prevalence"],
        lift=curve["average_precision"] / curve["prevalence"] if labels.sum() else None,
        calibration_z=float((labels.sum() - p.sum()) / np.sqrt(variance)),
        calibration_chi2_per_bin=float(chi2.mean()),
    )
    arrays = dict(
        logits=logits.astype(np.float32),
        labels=labels,
        precision=curve["precision"],
        recall=curve["recall"],
        logit_threshold=curve["logit_threshold"],
        **bins,
    )
    return metrics, arrays


def _evaluate_checkpoints(output, validation, episodes, prevalence, seed, device):
    """metrics.json, arrays and plots of best and final, as for voxel training.

    Voxel sets get the voxel-training metrics (and the last-layer predictive scores); the
    validation events get the event metrics under fidelity ``stream``.
    """
    from core.surrogates.experiment import _voxel_predictive
    from core.surrogates.readout import readout_metrics

    rows = []
    for checkpoint in ("best", "final"):
        saved = load_surrogate(output / "checkpoints" / checkpoint)
        saved.module.to(device)
        saved.module.eval()
        logits = []
        import torch

        with torch.no_grad():
            for start in range(0, len(validation), VALIDATION_CHUNK):
                logits.append(
                    saved.module.forward_events(validation[start : start + VALIDATION_CHUNK])
                    .float()
                    .cpu()
                )
        labels = event_labels(validation).cpu().numpy()
        summary, arrays = stream_metrics(
            labels, torch.cat(logits).numpy(), reference_rate=prevalence
        )
        rows.append(
            dict(
                checkpoint=checkpoint,
                step=saved.metadata.get("step"),
                split="validation",
                fidelity="stream",
                **summary,
            )
        )
        np.savez_compressed(output / f"stream_validation_{checkpoint}.npz", **arrays)
        for (split, fidelity), episode in episodes.items():
            summary, arrays = evaluate_surrogate(saved, episode.target, reference_rate=prevalence)
            summary.update(readout_metrics(episode.target, arrays, seed=seed))
            if getattr(saved, "last_layer", None) is not None:
                summary["voxel_predictive"] = _voxel_predictive(
                    saved, episode, output / f"{fidelity}_{split}_{checkpoint}_predictive"
                )
            rows.append(
                dict(
                    checkpoint=checkpoint,
                    step=saved.metadata.get("step"),
                    split=split,
                    fidelity=fidelity,
                    **summary,
                )
            )
            np.savez_compressed(output / f"{fidelity}_{split}_{checkpoint}.npz", **arrays)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    from viz.surrogate import plot_stream_run, plot_surrogate_run

    plot_surrogate_run(output)
    plot_stream_run(output)


def run_stream(config: StreamRunConfig) -> dict:
    import torch

    rank, world, local_rank = _distributed()
    training = config.training
    if training.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(f"cuda:{local_rank}" if training.device == "cuda" else "cpu")
    if world > 1:
        import torch.distributed as dist

        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        if device.type == "cuda":
            torch.cuda.set_device(device)
    output = Path(config.output_directory)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"{output} is not empty; choose a new output directory")
        output.mkdir(parents=True, exist_ok=True)
    compute = ComputeLog()
    loading = dict(config.event_source_options)
    train = open_event_source(
        config.event_source,
        config.event_store,
        "train",
        seed=training.seed,
        rank=rank,
        world_size=world,
        **loading,
    )
    normalization = train.normalization()
    torch.manual_seed(training.seed)
    model = build_surrogate(config.network, 3, config.dim_phi, seed=training.seed)
    model.module.set_stream_normalization(normalization)
    model.module.to(device)
    forward = _stream_module(model.module)
    if training.compile:
        forward = torch.compile(forward)
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel

        forward = DistributedDataParallel(
            forward, device_ids=[local_rank] if device.type == "cuda" else None
        )
    optimizer = torch.optim.Adam(model.module.parameters(), lr=training.learning_rate)

    def autocast():
        enabled = device.type == "cuda" and training.precision == "bf16"
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)

    # Training prevalence: constant-rate reference of the log-loss skill everywhere.
    prevalence = train.n_positive / train.n_events
    if not 0 < prevalence < 1:
        raise ValueError("Training events must contain events with label 1 and with label 0")
    validation, episodes, history_episode = None, {}, None
    with compute.stage("validation_loading"):
        if rank == 0:
            source = open_event_source(
                config.event_source, config.event_store, "validation", **loading
            )
            events = source.load()
            if training.validation_events is not None:
                events = events[: training.validation_events]
            validation = torch.as_tensor(np.ascontiguousarray(events), device=device)
            if config.evaluation:
                episodes, history_episode, data = _voxel_episodes(config)
                model.configure_normalization(data["normalization"])
    checkpoints = output / "checkpoints"
    history, best = [], dict(loss=math.inf, step=0)
    metadata = dict(
        resolved_config=config.model_dump(mode="json"),
        event_store=str(config.event_store),
        event_source=config.event_source,
        data_provider=config.data_provider,
        stream_normalization=normalization,
        training_events=train.n_events,
        training_prevalence=prevalence,
        validation_events=None if validation is None else len(validation),
        world_size=world,
        selection="bernoulli_log_loss on the validation events",
        uses_context=False,
        uncertainty=False,
    )
    if config.evaluation and rank == 0:
        metadata.update(voxel_evaluation=data["provenance"], history_voxels=data["history_voxels"])
        metadata["normalization.json"] = data["normalization"]
    model.metadata = metadata

    def evaluate(step, pass_index, train_loss):
        if rank != 0:
            return
        model.module.eval()
        loss = _log_loss(model.module.forward_events, validation, device, autocast)
        model.module.train()
        row = dict(step=step, pass_index=pass_index, training_loss=train_loss, stream_log_loss=loss)
        if history_episode is not None:
            row.update(
                evaluate_surrogate(model, history_episode.target, reference_rate=prevalence)[0]
            )
        history.append(row)
        print(f"step {step} pass {pass_index}: train {train_loss:.6f} val {loss:.6f}", flush=True)
        if loss < best["loss"]:
            best.update(loss=loss, step=step, pass_index=pass_index)
            model.metadata.update(best_step=step, best_stream_log_loss=loss)
            save_surrogate(checkpoints / "best", model, metadata={"step": step})

    step, events_seen, running = 0, 0, []
    model.module.train()
    with compute.stage("training"):
        for pass_index in range(training.n_passes):
            for batch in train.batches(
                training.batch_size, pass_index, drop_last=world > 1, device=device
            ):
                with autocast():
                    logits = forward(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), event_labels(batch)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                events_seen += len(batch)
                running.append(loss.detach())
                if step % training.eval_every == 0:
                    evaluate(step, pass_index, float(torch.stack(running).mean()))
                    running = []
            if running:
                evaluate(step, pass_index, float(torch.stack(running).mean()))
                running = []
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
    report = dict(steps=step, events_seen=events_seen * world, best=best, history=history)
    if rank == 0:
        model.metadata.update(steps=step, final_stream_log_loss=history[-1]["stream_log_loss"])
        save_surrogate(checkpoints / "final", model, metadata={"step": step})
        (output / "history.json").write_text(json.dumps(history, indent=1, allow_nan=False))
        if config.model.last_layer is not None:
            stream = open_event_source(config.event_source, config.event_store, "train", **loading)
            with compute.stage("last_layer"):
                for name in ("best", "final"):
                    selected = load_surrogate(checkpoints / name)
                    selected.module.to(device)
                    fit_stream_last_layer(
                        selected, stream, batch_size=training.batch_size, device=device
                    )
                    selected.module.cpu()
                    save_surrogate(checkpoints / name, selected)
        with compute.stage("evaluation"):
            _evaluate_checkpoints(
                output, validation, episodes, prevalence, config.evaluation_seed, device
            )
        (output / "experiment.json").write_text(
            json.dumps(dict(metadata, report=report), indent=2, allow_nan=False)
        )
        compute.write(output)
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()
    return report
