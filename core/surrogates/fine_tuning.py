"""Sequential representation training and natural-event output-layer fitting."""

from pathlib import Path

from core.surrogates.checkpoints import load_surrogate
from core.surrogates.training import FitResult, _fit_stage
from schemas.surrogates import NeuralTraining


def fit_two_stages(model, train, training, *, validation, selection, checkpoints):
    fine = training.fine_tuning
    if (fine.n_events or training.n_events) > train.n_events:
        raise ValueError("Fine-tuning n_events exceeds available training events per voxel")
    destination = None if checkpoints is None else Path(checkpoints)
    base = NeuralTraining.model_validate({**training.model_dump(), "fine_tuning": None})
    pre = _fit_stage(
        model,
        train,
        base,
        validation=validation,
        selection=selection,
        checkpoints=None if destination is None else destination / "stages/pretraining",
    )
    model.metadata["stage"] = "pretraining"
    if destination is not None:
        model.save(destination / "pretraining")
    head = model.output_layer()
    trainable = {id(p) for p in head.parameters()}
    names = [name for name, p in model.module.named_parameters() if id(p) in trainable]
    flags = {name: p.requires_grad for name, p in model.module.named_parameters()}
    natural = NeuralTraining.model_validate(
        {
            **base.model_dump(),
            "n_steps": fine.n_steps,
            "eval_every": fine.eval_every,
            "learning_rate": fine.learning_rate,
            "n_events": fine.n_events or base.n_events,
            "seed": base.seed + 1,
            "loss": "theory-truth" if model.config.kind == "legacy_cnp" else "bernoulli",
            "focal_gamma": 0,
            "sampling": {"strategy": "natural"},
            "weighting": {"strategy": "none"},
            "objective": {"strategy": "single"},
        }
    )
    model.metadata.update(
        stage="fine_tuning",
        fine_tuning={
            "trainable_parameters": names,
            "trainable_parameter_count": sum(p.numel() for p in head.parameters()),
            "pretraining_best_step": pre.best_step,
            "feature_mode": "eval",
            "settings": natural.model_dump(mode="json"),
        },
    )
    try:
        for p in model.module.parameters():
            p.requires_grad_(id(p) in trainable)
            p.grad = None
        post = _fit_stage(
            model,
            train,
            natural,
            validation=validation,
            selection=selection,
            checkpoints=destination,
            last_layer_only=True,
        )
    finally:
        for name, p in model.module.named_parameters():
            p.requires_grad_(flags[name])
        model.module.eval()
    history = [{**row, "stage": "pretraining", "stage_step": row["step"]} for row in pre.history]
    history += [
        {
            **row,
            "stage": "fine_tuning",
            "stage_step": row["step"],
            "step": base.n_steps + row["step"],
        }
        for row in post.history
    ]
    audit = {
        "strategy": "pretraining_then_last_layer",
        "pretraining": pre.sampling_audit,
        "fine_tuning": post.sampling_audit,
        "trainable_parameters": names,
    }
    model.metadata.update(
        training=training.model_dump(mode="json"),
        sampling_audit=audit,
        step=base.n_steps + post.best_step,
        stage_step=post.best_step,
    )
    if destination is not None:
        # Preserve the actual final weights while recording both-stage provenance.
        final = load_surrogate(destination / "final")
        final.metadata.update(
            training=training.model_dump(mode="json"),
            sampling_audit=audit,
            step=base.n_steps + fine.n_steps,
            stage_step=fine.n_steps,
        )
        final.save(destination / "final")
        model.save(destination / "best")
    return FitResult(base.n_steps + post.best_step, history, audit)
