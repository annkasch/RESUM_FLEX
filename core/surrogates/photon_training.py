"""Train a neural field on every photon of a photon store (``python -m core.surrogates photon``).

Each optimizer step takes one batch of raw photon records from the photon source (random
order, see its documentation); the field normalizes positions with the store's train
normalization (``BernoulliField.set_photon_normalization``) and encodes the direction from the
momentum. Checkpoints are selected by the Bernoulli log loss on the first
``validation_photons`` validation photons, held on the device. With ``last_layer`` the
selected checkpoints get a Laplace posterior from one streaming pass over all training
photons (exact Bernoulli Hessian of the final layer, as for voxel training).

Several GPUs: launch with ``torchrun``; every rank reads its own blocks of each pass and
DistributedDataParallel averages the gradients. Rank 0 validates and writes the outputs.

The checkpoints are ordinary field checkpoints: ``predict`` / ``score_event_model.py``
evaluate them on voxel data, mapping theta + offset into the field's photon units.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

from core.compute import ComputeLog
from core.photon_data import open_photon_source, photon_labels
from core.surrogates.checkpoints import save_surrogate
from core.surrogates.last_layer import _layer, last_layer_weights
from core.surrogates.models import build_surrogate
from schemas.surrogates import PhotonFieldRunConfig

VALIDATION_CHUNK = 2**20


def _distributed():
    """(rank, world size, local rank) from torchrun's environment; (0, 1, 0) otherwise."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, 0
    return int(os.environ["RANK"]), world, int(os.environ.get("LOCAL_RANK", "0"))


def _photon_module(field):
    import torch

    class PhotonForward(torch.nn.Module):
        """forward(records) = field.forward_photons(records), for DDP and torch.compile."""

        def __init__(self, field):
            super().__init__()
            self.field = field

        def forward(self, records):
            return self.field.forward_photons(records)

    return PhotonForward(field)


def _log_loss(forward, photons, device, autocast) -> float:
    """Mean Bernoulli log loss of photon records (on ``device``), in chunks, no gradients."""
    import torch

    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(photons), VALIDATION_CHUNK):
            chunk = photons[start : start + VALIDATION_CHUNK]
            with autocast():
                logits = forward(chunk)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), photon_labels(chunk), reduction="sum"
            )
            total += float(loss)
            count += len(chunk)
    return total / count


def fit_photon_last_layer(model, source, *, batch_size, device):
    """Laplace posterior of the final layer from one pass over all photons of ``source``."""
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
                field.forward_photons(batch)
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
        hessian="exact Bernoulli Hessian of the final layer over all training photons",
    )
    return posterior


def run_photon_field(config: PhotonFieldRunConfig) -> dict:
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
    loading = training.loading.model_dump()
    train = open_photon_source(
        config.photon_source,
        config.photon_store,
        "train",
        seed=training.seed,
        rank=rank,
        world_size=world,
        **loading,
    )
    normalization = train.normalization()
    torch.manual_seed(training.seed)
    model = build_surrogate(config.model, 3, config.dim_phi, seed=training.seed)
    model.module.set_photon_normalization(normalization)
    model.module.to(device)
    forward = _photon_module(model.module)
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

    validation = None
    with compute.stage("validation_loading"):
        if rank == 0:
            source = open_photon_source(
                config.photon_source, config.photon_store, "validation", **loading
            )
            photons = source.load()
            if training.validation_photons is not None:
                photons = photons[: training.validation_photons]
            validation = torch.as_tensor(np.ascontiguousarray(photons), device=device)
    prevalence = float(photon_labels(validation).mean()) if validation is not None else None
    constant_loss = (
        -(prevalence * math.log(prevalence) + (1 - prevalence) * math.log(1 - prevalence))
        if prevalence and 0 < prevalence < 1
        else None
    )
    checkpoints = output / "checkpoints"
    history, best = [], dict(loss=math.inf, step=0)
    metadata = dict(
        resolved_config=config.model_dump(mode="json"),
        photon_store=str(config.photon_store),
        photon_source=config.photon_source,
        data_provider=config.data_provider,
        photon_normalization=normalization,
        training_photons=train.n_photons,
        validation_photons=None if validation is None else len(validation),
        validation_constant_rate_log_loss=constant_loss,
        world_size=world,
        selection="validation bernoulli_log_loss",
        uses_context=False,
        uncertainty=False,
    )
    model.metadata = metadata

    def evaluate(step, pass_index, train_loss):
        if rank != 0:
            return
        model.module.eval()
        loss = _log_loss(model.module.forward_photons, validation, device, autocast)
        model.module.train()
        history.append(
            dict(step=step, pass_index=pass_index, train_loss=train_loss, validation_loss=loss)
        )
        print(f"step {step} pass {pass_index}: train {train_loss:.6f} val {loss:.6f}", flush=True)
        if loss < best["loss"]:
            best.update(loss=loss, step=step, pass_index=pass_index)
            model.metadata.update(best_step=step, best_validation_loss=loss)
            save_surrogate(checkpoints / "best", model)

    step, photons_seen, running = 0, 0, []
    model.module.train()
    with compute.stage("training"):
        for pass_index in range(training.n_passes):
            for batch in train.batches(
                training.batch_size, pass_index, drop_last=world > 1, device=device
            ):
                with autocast():
                    logits = forward(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), photon_labels(batch)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                photons_seen += len(batch)
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
    report = dict(steps=step, photons_seen=photons_seen * world, best=best, history=history)
    if rank == 0:
        model.metadata.update(steps=step, final_validation_loss=history[-1]["validation_loss"])
        save_surrogate(checkpoints / "final", model)
        if config.model.last_layer is not None:
            from core.surrogates.checkpoints import load_surrogate

            stream = open_photon_source(
                config.photon_source, config.photon_store, "train", **loading
            )
            with compute.stage("last_layer"):
                for name in ("best", "final"):
                    selected = load_surrogate(checkpoints / name)
                    selected.module.to(device)
                    fit_photon_last_layer(
                        selected, stream, batch_size=training.batch_size, device=device
                    )
                    selected.module.cpu()
                    save_surrogate(checkpoints / name, selected)
        (output / "history.json").write_text(json.dumps(history, indent=1))
        (output / "experiment.json").write_text(
            json.dumps(dict(metadata, report=report), indent=2, allow_nan=False)
        )
        compute.write(output)
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()
    return report
