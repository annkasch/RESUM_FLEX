"""Separate neural and histogram-tree fitting behind the component API."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.surrogates.evaluation import evaluate_surrogate
from schemas.surrogates import SurrogateConfig


@dataclass
class FitResult:
    best_step: int
    history: list[dict]
    sampling_audit: dict


def fit_surrogate(
    model, train, training, *, validation=None, selection="voxel_rate_mae", checkpoints=None
):
    config = SurrogateConfig(model=model.config, training=training, selection=selection)
    training = config.training
    # Training batches must be real, normalized inputs with binary labels.
    model.validate(train, train if model.uses_context else None)
    if not np.isin(train.labels, [0, 1]).all():
        raise ValueError("Training requires real binary labels")
    if validation is not None:
        model.validate(validation.target, validation.context)
        if selection == "average_precision" and not np.any(validation.target.labels):
            raise ValueError("Cannot select by AP when validation contains no positives")
    history, best_state = [], None
    best_score = -np.inf if selection == "average_precision" else np.inf
    best_step = 0
    destination = None if checkpoints is None else Path(checkpoints)
    model.metadata.update(
        training=training.model_dump(mode="json"),
        selection=selection,
        uses_context=model.uses_context,
        uncertainty=False,
    )

    def snapshot():
        if training.backend == "bdt":
            return deepcopy(model.estimator)
        return {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}

    def observe(step, loss=None):
        nonlocal best_score, best_step, best_state
        metrics = (
            {}
            if validation is None
            else evaluate_surrogate(model, validation.target, context=validation.context)[0]
        )
        row = dict(step=step, **metrics)
        if loss is not None:
            row["training_loss"] = loss
        history.append(row)
        score = metrics.get(selection)
        better = validation is None or (
            score is not None
            and (score > best_score if selection == "average_precision" else score < best_score)
        )
        if better:
            best_score, best_step, best_state = score, step, snapshot()
            if destination:
                model.save(destination / "best", metadata={"step": step})

    if training.backend == "bdt":
        if len(np.unique(train.labels)) != 2:
            raise ValueError("BDT training requires both binary classes")
        from core.surrogate_bdt import event_features

        cfg = model.config.architecture
        x, y = event_features(train), train.labels.ravel()
        w = training.weighting
        weights = (
            np.where(y == 1, w.positive, w.negative) if w.strategy == "class_weights" else None
        )
        # Reset on each fit call; warm_start is only for incremental checkpoints.
        from core.surrogate_bdt import build_bdt

        model.estimator = build_bdt(cfg)
        counts = list(range(cfg.eval_every, cfg.max_iter + 1, cfg.eval_every))
        if not counts or counts[-1] != cfg.max_iter:
            counts.append(cfg.max_iter)
        for step in counts:
            model.estimator.set_params(max_iter=step)
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=1):
                model.estimator.fit(x, y, sample_weight=weights)
            observe(step)
        last_step = cfg.max_iter
        audit = dict(
            strategy="all_natural_events",
            events=int(y.size),
            positives=int(y.sum()),
            positive_fraction=float(y.mean()),
        )
    else:
        import torch

        from core.binary_losses import binary_focal_loss_with_logits
        from core.target_sampling import RealTargetSampler

        if training.n_events > train.n_events:
            raise ValueError("n_events exceeds available training events per voxel")
        if training.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.manual_seed(training.seed)
        model.module.to(training.device).train()
        optimizer = torch.optim.Adam(model.module.parameters(), lr=training.learning_rate)
        sampler = RealTargetSampler(
            train,
            seed=training.seed,
            batch_size=training.batch_size,
            n_events=training.n_events,
            n_context_min=training.n_context_min,
            n_context_max=training.n_context_max,
            positive_fraction=training.sampling.positive_fraction,
        )
        positives, events, losses = 0, 0, []
        for step in range(1, training.n_steps + 1):
            context, target, weights = sampler.next()
            positives += int(target.labels.sum())
            events += int(target.labels.size)
            w = training.weighting
            if w.strategy == "class_weights":
                weights = np.where(target.labels == 1, w.positive, w.negative)
            logits = model.module(context, target)
            loss = binary_focal_loss_with_logits(
                logits, target.labels, gamma=training.focal_gamma, weights=weights
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at step {step}")
            optimizer.zero_grad()
            loss.backward()
            if training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), training.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % training.eval_every == 0 or step == training.n_steps:
                observe(step, float(np.mean(losses)))
                losses.clear()
        last_step = training.n_steps
        audit = dict(
            strategy=training.sampling.strategy,
            events=events,
            positives=positives,
            positive_fraction=positives / events,
            requested_positive_fraction=training.sampling.positive_fraction,
        )
    model.metadata["sampling_audit"] = audit
    if destination:
        model.save(destination / "final", metadata={"step": last_step})
    if best_state is None:
        raise ValueError("No valid selection score was produced")
    if training.backend == "bdt":
        model.estimator = best_state
    else:
        model.module.load_state_dict(best_state)
        model.module.eval()
    model.metadata["step"] = best_step
    if destination:
        model.save(destination / "best")
    return FitResult(best_step, history, audit)
