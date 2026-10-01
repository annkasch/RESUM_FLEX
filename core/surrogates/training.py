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
    if training.backend == "neural" and training.fine_tuning is not None:
        from core.surrogates.fine_tuning import fit_two_stages

        return fit_two_stages(
            model,
            train,
            training,
            validation=validation,
            selection=selection,
            checkpoints=checkpoints,
        )
    return _fit_stage(
        model, train, training, validation=validation, selection=selection, checkpoints=checkpoints
    )


def _fit_stage(
    model,
    train,
    training,
    *,
    validation=None,
    selection="voxel_rate_mae",
    checkpoints=None,
    last_layer_only=False,
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
        prediction_scale="legacy_proxy" if model.config.kind == "legacy_cnp" else None,
    )

    def snapshot():
        if training.backend == "bdt":
            return deepcopy(model.estimator)
        return {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}

    def observe(step, loss=None, components=None):
        nonlocal best_score, best_step, best_state
        metrics = (
            {}
            if validation is None
            else evaluate_surrogate(model, validation.target, context=validation.context)[0]
        )
        row = dict(step=step, **metrics)
        if loss is not None:
            row["training_loss"] = loss
        if components is not None:
            row.update(components)
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

        def event_loss(output, labels, weights=None):
            if model.config.kind == "legacy_cnp":
                from core.surrogate_cnp import cnp_loss

                labels = torch.as_tensor(
                    labels, dtype=output.mu_logit.dtype, device=output.mu_logit.device
                )
                return cnp_loss(output, labels, objective=training.loss)
            return binary_focal_loss_with_logits(
                output, labels, gamma=training.focal_gamma, weights=weights
            )

        if training.n_events > train.n_events:
            raise ValueError("n_events exceeds available training events per voxel")
        if training.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.manual_seed(training.seed)
        model.module.to(training.device)
        # Frozen features must be deterministic: dropout stays disabled in stage 2.
        model.module.train(not last_layer_only)
        parameters = [p for p in model.module.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=training.learning_rate)
        mixup = training.sampling.strategy == "class_aware_mixup"
        sampler_args = dict(
            seed=training.seed,
            batch_size=training.batch_size,
            n_events=training.n_events,
        )
        if mixup:
            from core.mixup import ClassAwareMixupSource

            sampler = ClassAwareMixupSource(
                train,
                **sampler_args,
                n_context=training.n_context_min,
                alpha=training.sampling.mixup.alpha,
                mix_context=training.sampling.mixup.mix_context,
            )
            size_rng = np.random.default_rng(np.random.SeedSequence([training.seed, 2]))
        else:
            sampler = RealTargetSampler(
                train,
                **sampler_args,
                n_context_min=training.n_context_min,
                n_context_max=training.n_context_max,
                positive_fraction=training.sampling.positive_fraction,
            )
        combined = training.objective.strategy == "real_plus_mixup"
        real_events, real_positives = 0, 0
        real_losses, mixup_losses = [], []
        positives, events, losses = 0, 0, []
        label_mass, soft_events = 0.0, 0
        for step in range(1, training.n_steps + 1):
            if mixup:
                nc = int(size_rng.integers(training.n_context_min, training.n_context_max + 1))
                if combined:
                    context, target, real_target = sampler.next(
                        n_context=nc, real_target_ratio=training.objective.real_target_ratio
                    )
                    real_events += int(real_target.labels.size)
                    real_positives += int(real_target.labels.sum())
                else:
                    context, target = sampler.next(n_context=nc)
                weights = None
                label_mass += float(target.labels.sum())
                soft_events += int(((target.labels > 0) & (target.labels < 1)).sum())
            else:
                context, target, weights = sampler.next()
                positives += int(target.labels.sum())
            events += int(target.labels.size)
            w = training.weighting
            if w.strategy == "class_weights":
                weights = np.where(target.labels == 1, w.positive, w.negative)
            logits = model.module(context, target)
            loss = event_loss(logits, target.labels, weights)
            if combined:
                mixup_loss = loss
                real_logits = model.module(context, real_target)
                real_loss = event_loss(real_logits, real_target.labels)
                loss = real_loss + training.objective.mixup_loss_weight * mixup_loss
                real_losses.append(float(real_loss.detach()))
                mixup_losses.append(float(mixup_loss.detach()))
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at step {step}")
            optimizer.zero_grad()
            loss.backward()
            if training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), training.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % training.eval_every == 0 or step == training.n_steps:
                components = None
                if combined:
                    suffix = "nll" if training.loss == "practice-truth" else "bce"
                    components = {
                        f"training_real_{suffix}": float(np.mean(real_losses)),
                        f"training_mixup_{suffix}": float(np.mean(mixup_losses)),
                    }
                    real_losses.clear()
                    mixup_losses.clear()
                observe(step, float(np.mean(losses)), components)
                losses.clear()
        last_step = training.n_steps
        audit = dict(
            strategy=training.sampling.strategy,
            events=events,
            positives=positives,
            positive_fraction=positives / events,
            requested_positive_fraction=training.sampling.positive_fraction,
        )
        if mixup:
            audit = dict(
                strategy=training.sampling.strategy,
                events=events,
                label_mass=label_mass,
                mean_target_label=label_mass / events,
                soft_label_events=soft_events,
                alpha=training.sampling.mixup.alpha,
                mix_context=training.sampling.mixup.mix_context,
                source_split="fresh_per_batch",
                weighting="none",
            )
        if combined:
            audit.update(
                objective="real_plus_mixup",
                real_target_events=real_events,
                real_target_positives=real_positives,
                real_target_positive_fraction=real_positives / real_events,
                mixed_target_events=events,
                total_target_predictions=real_events + events,
                mixup_loss_weight=training.objective.mixup_loss_weight,
                real_target_ratio=training.objective.real_target_ratio,
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
