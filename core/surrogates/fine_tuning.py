"""Compatibility adapter: old fine_tuning settings become ordinary ordered stages."""

from pathlib import Path

from core.surrogates.checkpoints import load_surrogate
from core.surrogates.stages import fit_stages
from schemas.surrogates import NeuralTraining, TrainingStage


def fine_tuning_stages(training, kind):
    fine = training.fine_tuning
    base = NeuralTraining.model_validate({**training.model_dump(), "fine_tuning": None})
    natural = NeuralTraining.model_validate(
        {
            **base.model_dump(),
            "n_steps": fine.n_steps,
            "eval_every": fine.eval_every,
            "learning_rate": fine.learning_rate,
            "n_events": fine.n_events or base.n_events,
            "seed": base.seed + 1,
            "loss": "theory-truth" if kind == "legacy_cnp" else "bernoulli",
            "focal_gamma": 0,
            "sampling": {"strategy": "natural"},
            "weighting": {"strategy": "none"},
            "objective": {"strategy": "single"},
        }
    )
    return [
        TrainingStage(name="pretraining", training=base),
        TrainingStage(
            name="fine_tuning",
            start_from="pretraining/best",
            trainable="output_layer",
            training=natural,
        ),
    ]


def fit_two_stages(model, train, training, *, validation, selection, checkpoints):
    if (training.fine_tuning.n_events or training.n_events) > train.n_events:
        raise ValueError("Fine-tuning n_events exceeds available training events per voxel")
    stages = fine_tuning_stages(training, model.config.kind)
    for stage in stages:
        stage.selection = selection
    result = fit_stages(model, train, stages, validation=validation, checkpoints=checkpoints)
    detail = model.metadata["stage_details"]["fine_tuning"]
    names = detail["trainable_parameters"]
    fine_metadata = dict(
        trainable_parameters=names,
        trainable_parameter_count=sum(
            p.numel() for n, p in model.module.named_parameters() if n in names
        ),
        pretraining_best_step=model.metadata["stage_details"]["pretraining"]["selected_step"],
        feature_mode="eval",
        settings=detail["settings"],
    )
    audits = result.sampling_audit["stages"]
    audit = dict(strategy="pretraining_then_last_layer", **audits, trainable_parameters=names)
    result.sampling_audit = audit
    model.metadata.update(
        training=training.model_dump(mode="json"), fine_tuning=fine_metadata, sampling_audit=audit
    )
    if checkpoints is not None:
        dest = Path(checkpoints)
        final = load_surrogate(dest / "final")
        final.metadata.update(
            training=training.model_dump(mode="json"),
            fine_tuning=fine_metadata,
            sampling_audit=audit,
        )
        final.save(dest / "final")
        model.save(dest / "best")
    return result
