"""Ordered training stages with explicit initialization and parameter scopes."""

from copy import deepcopy
from pathlib import Path

from core.surrogates.training import FitResult, _fit_stage


def fit_stages(model, train, stages, *, validation=None, checkpoints=None):
    destination = None if checkpoints is None else Path(checkpoints)
    history, audits, details, states = [], {}, {}, {}
    offset = 0

    def capture():
        if model.config.kind == "bdt":
            return deepcopy(model.estimator)
        return {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}

    def restore(state):
        if model.config.kind == "bdt":
            model.estimator = deepcopy(state)
        else:
            model.module.load_state_dict(state)

    initial = capture()
    for stage in stages:
        if stage.start_from == "initial":
            restore(initial)
        elif stage.start_from in states:
            restore(states[stage.start_from])
        else:
            raise ValueError(f"Unknown stage dependency: {stage.start_from}")
        frozen = stage.trainable == "output_layer"
        flags, names = {}, []
        if stage.training.backend == "neural":
            flags = {n: p.requires_grad for n, p in model.module.named_parameters()}
            allowed = {id(p) for p in model.output_layer().parameters()} if frozen else None
            for name, param in model.module.named_parameters():
                param.requires_grad_(allowed is None or id(param) in allowed)
                param.grad = None
                if param.requires_grad:
                    names.append(name)
        elif frozen:
            raise ValueError("BDT does not support output-layer fine-tuning")
        model.metadata["stage"] = stage.name
        folder = None if destination is None else destination / "stages" / stage.name
        # Retain the final state before _fit_stage restores its selected best state.
        # A temporary checkpoint directory supports final dependencies for in-memory fits.
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            stage_folder = folder or Path(temporary)
            try:
                result = _fit_stage(
                    model,
                    train,
                    stage.training,
                    validation=validation,
                    selection=stage.selection,
                    checkpoints=stage_folder,
                    last_layer_only=frozen,
                )
                from core.surrogates.checkpoints import load_surrogate

                final = load_surrogate(stage_folder / "final")
                states[stage.name + "/best"] = capture()
                states[stage.name + "/final"] = (
                    deepcopy(final.estimator)
                    if stage.training.backend == "bdt"
                    else {k: v.detach().cpu().clone() for k, v in final.module.state_dict().items()}
                )
            finally:
                if flags:
                    for name, param in model.module.named_parameters():
                        param.requires_grad_(flags[name])
                    model.module.eval()
        steps = (
            stage.training.n_steps
            if stage.training.backend == "neural"
            else model.config.architecture.max_iter
        )
        details[stage.name] = dict(
            start_from=stage.start_from,
            trainable=stage.trainable,
            trainable_parameters=names,
            selected_step=result.best_step,
            settings=stage.training.model_dump(mode="json"),
        )
        history.extend(
            {**r, "stage": stage.name, "stage_step": r["step"], "step": offset + r["step"]}
            for r in result.history
        )
        audits[stage.name] = result.sampling_audit
        selected_step = offset + result.best_step
        offset += steps
        if destination:
            # Named selected checkpoints remain useful to old notebook consumers.
            model.save(destination / stage.name)
    audit = {"strategy": "ordered_stages", "stages": audits}
    model.metadata.update(
        stage_details=details, sampling_audit=audit, step=selected_step, stage_step=result.best_step
    )
    if destination:
        final.metadata.update(
            stage_details=details, sampling_audit=audit, step=offset, stage_step=steps
        )
        final.save(destination / "final")
        model.save(destination / "best")
    return FitResult(selected_step, history, audit)
