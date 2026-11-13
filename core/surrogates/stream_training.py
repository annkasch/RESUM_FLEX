"""Train an event model on every event of an event store (``python -m core.surrogates stream``).

Neural kinds (field; mlp and transformer as the unencoded field with that network): each
optimizer step takes one batch of raw event records from the event source (random
order, see its documentation); the field normalizes positions with the store's train
normalization (``BernoulliField.set_stream_normalization``) and encodes the direction from the
momentum. Checkpoints are selected by the Bernoulli log loss on the first
``validation_events`` validation events, held on the device. With ``last_layer`` the
selected checkpoints get a Laplace posterior from one streaming pass over all training
events (exact Bernoulli Hessian of the final layer, as for design point training).

Several GPUs: launch with ``torchrun``; every rank reads its own blocks of each pass and
DistributedDataParallel averages the gradients. Rank 0 validates and writes the outputs.

``kind: bdt``: boosted trees do not learn batch by batch, so one pass over the training
events keeps a case-control sample in memory (every positive, a seeded fraction of the
negatives weighted 1 / fraction; ``case_control_sample``); trees are added ``eval_every`` at a
time and selected by the same validation log loss. No last layer.

The checkpoints are ordinary field (or BDT) checkpoints: ``predict`` / ``score_event_model.py``
evaluate them on design point data, mapping theta + offset into the model's stream units
(trees: physical record columns). There is no stream CNP (docs/stream.md).
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
from core.surrogates.evaluation import (
    CALIBRATION_BINS,
    calibration_bins,
    evaluate_surrogate,
    feature_calibration,
)
from core.surrogates.last_layer import _layer, last_layer_weights
from core.surrogates.models import build_surrogate
from core.surrogates.readout import NULL_DRAWS
from schemas.surrogates import StreamRunConfig

VALIDATION_CHUNK = 2**20
# Sub-stream of the training seed for the case-control sample of a stream BDT.
SAMPLE_KEY = 7


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


def stream_logits(model, records):
    """Logits [M] float32 (CPU) of raw records [M, W + 1] (torch) for any stream-trained model."""
    import torch

    if model.config.kind == "bdt":
        features = records[:, :-1].cpu().numpy().astype(np.float64)  # [M, W]
        return torch.as_tensor(model.estimator.decision_function(features), dtype=torch.float32)
    with torch.no_grad():
        return torch.cat(
            [
                model.module.forward_events(records[start : start + VALIDATION_CHUNK]).float().cpu()
                for start in range(0, len(records), VALIDATION_CHUNK)
            ]
        )


def case_control_sample(source, *, negatives_per_positive, batch_size, seed):
    """In-memory case-control sample of an event source, from one pass over it.

    Keeps every positive and each negative with probability q = min(1, k n+ / n-), k =
    ``negatives_per_positive``; kept negatives get weight 1 / q, so weighted sums over the
    sample estimate sums over all events (calibrated probabilities). Returns (records
    [K, W + 1] float32 numpy, weights [K] float64, summary dict).
    """
    negatives = source.n_events - source.n_positive
    if source.n_positive == 0 or negatives == 0:
        raise ValueError("Training events must contain events with label 1 and with label 0")
    fraction = min(1.0, negatives_per_positive * source.n_positive / negatives)
    rng = np.random.default_rng(np.random.SeedSequence([seed, SAMPLE_KEY]))
    kept, labels = [], []
    for batch in source.batches(batch_size, 0):
        positive = event_labels(batch).cpu().numpy() == 1
        keep = positive | (rng.random(len(positive)) < fraction)
        kept.append(batch.cpu().numpy()[keep])
        labels.append(positive[keep])
    records, positive = np.concatenate(kept), np.concatenate(labels)
    weights = np.where(positive, 1.0, 1.0 / fraction)
    summary = dict(
        strategy="case_control",
        events=int(len(records)),
        positives=int(positive.sum()),
        negatives=int((~positive).sum()),
        negative_fraction=fraction,
        negative_weight=1.0 / fraction,
        source_events=int(source.n_events),
        source_positives=int(source.n_positive),
    )
    return records, weights, summary


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


def _design_point_episodes(config: StreamRunConfig):
    """Labelled design point episodes of ``config.evaluation`` (context-free: no context events).

    Returns (episodes {(split, fidelity): Episode}, the history episode, data record).
    """
    from core.gp_subsample import subsample_design_points
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
    history_episode, history_design_points = None, None
    if first is not None:
        target = episodes["validation", first.fidelity].target
        index = subsample_design_points(
            target.batch_size, config.history_design_points, config.evaluation_seed, 0
        )
        history_episode = Episode(
            None, target.take(index) if len(index) < target.batch_size else target
        )
        history_design_points = dict(
            set=f"validation/{first.fidelity}", available=target.batch_size, used=len(index)
        )
    data = dict(
        normalization=provider.normalization(),
        provenance=provenance,
        history_design_points=history_design_points,
    )
    return episodes, history_episode, data


def stream_metrics(
    labels,
    logits,
    *,
    reference_rate,
    features=None,
    residual=None,
    seed=0,
    feature_bins=CALIBRATION_BINS,
):
    """Event metrics of events without design-point structure, as in design point evaluation.

    labels, logits: [M] numpy; features: optional [M, W] record input columns, which add
    the calibration per record column (``record_j``, ``feature_bins`` bins) and, with ``residual``
    (ResidualTestConfig), the residual learnability test split by event. Returns (metrics,
    arrays) with calibration bins and the precision-recall curve.
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
    if features is not None:
        features = np.asarray(features, dtype=np.float64)  # [M, W]
        names = [f"record_{j}" for j in range(features.shape[1])]
        metrics["feature_calibration"], feature_arrays = feature_calibration(
            names, features, labels, p, feature_bins
        )
        arrays.update(feature_arrays)
    if residual is not None:
        if features is None:
            raise ValueError("The residual test needs the record features")
        from core.surrogates.residual import stream_residual

        metrics["residual_test"], residual_arrays = stream_residual(
            features, labels, logits, residual, seed=seed, bins=feature_bins
        )
        arrays.update(residual_arrays)
    return metrics, arrays


def _evaluate_checkpoints(
    output,
    validation,
    episodes,
    prevalence,
    seed,
    device,
    residual=None,
    feature_bins=CALIBRATION_BINS,
    null_draws=NULL_DRAWS,
    normalization=None,
):
    """metrics.json, arrays and plots of best and final, as for design point training.

    Design point sets get the metrics of design point training (and the last-layer predictive
    scores); the validation events get the event metrics under fidelity ``stream``.
    """
    from core.surrogates.experiment import _design_point_predictive
    from core.surrogates.readout import readout_metrics

    rows = []
    for checkpoint in ("best", "final"):
        saved = load_surrogate(output / "checkpoints" / checkpoint)
        if saved.config.kind == "bdt":
            if episodes:
                saved.configure_normalization(normalization)
        else:
            saved.module.to(device)
            saved.module.eval()
        labels = event_labels(validation).cpu().numpy()
        summary, arrays = stream_metrics(
            labels,
            stream_logits(saved, validation).numpy(),
            reference_rate=prevalence,
            features=validation[:, :-1].float().cpu().numpy(),
            residual=residual,
            seed=seed,
            feature_bins=feature_bins,
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
            readout, readout_arrays = readout_metrics(
                episode.target,
                arrays,
                seed=seed,
                residual=residual,
                feature_bins=feature_bins,
                null_draws=null_draws,
            )
            summary.update(readout)
            arrays.update(readout_arrays)
            if getattr(saved, "last_layer", None) is not None:
                summary["design_point_predictive"] = _design_point_predictive(
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


def _new_output(config: StreamRunConfig) -> Path:
    output = Path(config.output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validation_events(config: StreamRunConfig, loading: dict, device):
    """The first ``validation_events`` validation records as a tensor on ``device``."""
    import torch

    source = open_event_source(config.event_source, config.event_store, "validation", **loading)
    events = source.load()
    if config.training.validation_events is not None:
        events = events[: config.training.validation_events]
    return torch.as_tensor(np.ascontiguousarray(events), device=device)


def _run_stream_trees(config: StreamRunConfig) -> dict:
    """``kind: bdt``: boosted trees on a case-control sample, selected on validation events."""
    import torch

    from core.surrogate_bdt import build_bdt

    if _distributed()[1] > 1:
        raise ValueError("A stream BDT trains in one process; launch it without torchrun")
    if config.residual_test is not None:
        from core.surrogates.residual import require_lightgbm

        require_lightgbm()
    training, trees = config.training, config.model.architecture
    output = _new_output(config)
    compute = ComputeLog()
    loading = dict(config.event_source_options)
    train = open_event_source(
        config.event_source,
        config.event_store,
        "train",
        seed=training.seed,
        rank=0,
        world_size=1,
        **loading,
    )
    normalization = train.normalization()
    prevalence = train.n_positive / train.n_events
    with compute.stage("sampling"):
        records, weights, sample = case_control_sample(
            train,
            negatives_per_positive=training.negatives_per_positive,
            batch_size=training.batch_size,
            seed=training.seed,
        )
    x = records[:, :-1].astype(np.float64)  # [K, W]
    y = (event_labels(torch.as_tensor(records)).numpy() == 1).astype(np.int8)
    del records
    model = build_surrogate(config.model, config.dim_theta, config.dim_phi, seed=training.seed)
    model.stream_position = config.position_columns
    if x.shape[1] != model.n_features:
        raise ValueError(
            f"Stream records have {x.shape[1]} input columns; dim_theta + dim_phi - "
            f"position columns = {model.n_features}"
        )
    episodes, history_episode, data = {}, None, None
    with compute.stage("validation_loading"):
        validation = _validation_events(config, loading, "cpu")
        if config.evaluation:
            episodes, history_episode, data = _design_point_episodes(config)
            model.configure_normalization(data["normalization"])
    checkpoints = output / "checkpoints"
    metadata = dict(
        resolved_config=config.model_dump(mode="json"),
        event_store=str(config.event_store),
        event_source=config.event_source,
        data_provider=config.data_provider,
        stream_normalization=normalization,
        training_events=train.n_events,
        training_prevalence=prevalence,
        training_sample=sample,
        validation_events=len(validation),
        world_size=1,
        selection="bernoulli_log_loss on the validation events",
        uses_context=False,
        uncertainty=False,
    )
    if data is not None:
        metadata.update(
            design_point_evaluation=data["provenance"],
            history_design_points=data["history_design_points"],
        )
        metadata["normalization.json"] = data["normalization"]
    model.metadata = metadata
    labels = event_labels(validation).numpy()
    model.estimator = build_bdt(trees)
    counts = list(range(trees.eval_every, trees.max_iter + 1, trees.eval_every))
    if not counts or counts[-1] != trees.max_iter:
        counts.append(trees.max_iter)
    history, best = [], dict(loss=math.inf, step=0)
    with compute.stage("training"):
        for step in counts:
            model.estimator.set_params(max_iter=step)
            model.estimator.fit(x, y, sample_weight=weights)  # warm start: adds trees
            train_logits = model.estimator.decision_function(x)
            train_loss = float(
                np.average(
                    np.where(y == 1, np.logaddexp(0, -train_logits), np.logaddexp(0, train_logits)),
                    weights=weights,
                )
            )
            logits = stream_logits(model, validation).double().numpy()
            loss = float(
                np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits)).mean()
            )
            row = dict(step=step, pass_index=0, training_loss=train_loss, stream_log_loss=loss)
            if history_episode is not None:
                row.update(
                    evaluate_surrogate(model, history_episode.target, reference_rate=prevalence)[0]
                )
            history.append(row)
            print(f"trees {step}: train {train_loss:.6f} val {loss:.6f}", flush=True)
            if loss < best["loss"]:
                best.update(loss=loss, step=step, pass_index=0)
                model.metadata.update(best_step=step, best_stream_log_loss=loss)
                save_surrogate(checkpoints / "best", model, metadata={"step": step})
    model.metadata.update(steps=counts[-1], final_stream_log_loss=history[-1]["stream_log_loss"])
    save_surrogate(checkpoints / "final", model, metadata={"step": counts[-1]})
    (output / "history.json").write_text(json.dumps(history, indent=1, allow_nan=False))
    report = dict(steps=counts[-1], events_seen=sample["events"], best=best, history=history)
    with compute.stage("evaluation"):
        _evaluate_checkpoints(
            output,
            validation,
            episodes,
            prevalence,
            config.evaluation_seed,
            "cpu",
            residual=config.residual_test,
            feature_bins=config.feature_calibration.bins,
            null_draws=config.within_point_null_draws,
            normalization=None if data is None else data["normalization"],
        )
    (output / "experiment.json").write_text(
        json.dumps(dict(metadata, report=report), indent=2, allow_nan=False)
    )
    compute.write(output)
    return report


def run_stream(config: StreamRunConfig) -> dict:
    if config.model.kind == "bdt":
        return _run_stream_trees(config)
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
    if config.residual_test is not None:
        from core.surrogates.residual import require_lightgbm

        require_lightgbm()
    output = _new_output(config) if rank == 0 else Path(config.output_directory)
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
    model = build_surrogate(config.network, config.dim_theta, config.dim_phi, seed=training.seed)
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
            validation = _validation_events(config, loading, device)
            if config.evaluation:
                episodes, history_episode, data = _design_point_episodes(config)
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
        metadata.update(
            design_point_evaluation=data["provenance"],
            history_design_points=data["history_design_points"],
        )
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
        if config.network.last_layer is not None:
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
                output,
                validation,
                episodes,
                prevalence,
                config.evaluation_seed,
                device,
                residual=config.residual_test,
                feature_bins=config.feature_calibration.bins,
                null_draws=config.within_point_null_draws,
            )
        (output / "experiment.json").write_text(
            json.dumps(dict(metadata, report=report), indent=2, allow_nan=False)
        )
        compute.write(output)
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()
    return report
