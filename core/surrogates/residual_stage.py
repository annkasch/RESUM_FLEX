"""Residual learnability test on a finished run (``python -m core.surrogates residual``).

Reads the logits and labels the run saved for each held-out set and checkpoint
(``{fidelity}_{split}_{checkpoint}.npz``, ``stream_validation_{checkpoint}.npz``), reloads
the inputs from the run's resolved config and runs ``core.surrogates.residual`` with the
run's own evaluation seed, so the result equals the read-out of a run trained with the
same ``residual_test:`` settings. Reloaded inputs must match the run: provider provenance
(e.g. file hashes) and labels. The run directory is never written to.
"""

import json
from pathlib import Path

import numpy as np

from core.surrogates.residual import design_point_residual, require_lightgbm, stream_residual
from schemas.surrogates import (
    ResidualRunConfig,
    ResidualTestConfig,
    StreamRunConfig,
    SurrogateRunConfig,
)


def _same_labels(saved, loaded, name):
    if saved.shape != loaded.shape or not np.array_equal(saved, loaded):
        raise ValueError(f"{name}: reloaded labels differ from the run's saved labels")


def _check_provenance(recorded, loaded, name):
    if recorded is None:
        raise ValueError(f"{name}: the run recorded no provenance for this set")
    if recorded != loaded:
        keys = sorted(k for k in {*recorded, *loaded} if recorded.get(k) != loaded.get(k))
        raise ValueError(f"{name}: data changed since the run (differs in {keys})")


def _held_out_sets(run):
    """(split, fidelity) of every held-out set in the run's metrics.json, in order."""
    rows = json.loads((run / "metrics.json").read_text())
    keys = [(r["split"], r["fidelity"]) for r in rows if r["split"] != "train"]
    return list(dict.fromkeys(keys))


def _training_run_targets(run, experiment, sets):
    """Target batches of a design-point training run, as its evaluation built them."""
    from core.surrogate_cnp import split_context_target
    from core.surrogates.datasets import load_provider

    config = SurrogateRunConfig.model_validate(experiment["resolved_config"])
    provider = load_provider(config.data_provider, config.data_directory)
    test_file = run / "test_evaluation.json"
    tests = json.loads(test_file.read_text()).get("data", {}) if test_file.exists() else {}
    targets = {}
    for split, fidelity in sets:
        name = f"{split}/{fidelity}"
        batch, provenance = provider.load(config.partitions[split], fidelity)
        recorded = tests.get(fidelity) if split == "test" else experiment["data"].get(name)
        _check_provenance(recorded, provenance, name)
        _, targets[split, fidelity] = split_context_target(
            batch,
            config.validation_context_events,
            seed=config.validation_seed,
            n_pool=config.validation_context_pool,
        )
    return targets, config.validation_seed


def _stream_run_inputs(experiment, sets):
    """Design-point batches and the validation records of a stream run."""
    from core.surrogates.datasets import load_provider

    config = StreamRunConfig.model_validate(experiment["resolved_config"])
    targets, records = {}, None
    grouped = [key for key in sets if key[1] != "stream"]
    if grouped:
        provider = load_provider(config.data_provider, config.data_directory)
        partitions = {(e.split, e.fidelity): e.partition for e in config.evaluation}
        for split, fidelity in grouped:
            name = f"{split}/{fidelity}"
            targets[split, fidelity], provenance = provider.load(
                partitions[split, fidelity], fidelity
            )
            provenance["partition"] = partitions[split, fidelity]
            _check_provenance(experiment.get("voxel_evaluation", {}).get(name), provenance, name)
    if len(grouped) < len(sets):
        from core.event_stream import open_event_source

        source = open_event_source(
            config.event_source,
            config.event_store,
            "validation",
            **config.event_source_options,
        )
        records = np.ascontiguousarray(source.load()[: experiment["validation_events"]])
    return targets, records, config.evaluation_seed


def run_residual(config: ResidualRunConfig) -> Path:
    """residual_test.json and one arrays file per (set, checkpoint) in output_directory."""
    output = config.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_directory; {output} is not empty")
    require_lightgbm()
    run = config.run
    experiment = json.loads((run / "experiment.json").read_text())
    available = _held_out_sets(run)
    if config.sets is None:
        sets = available
    else:
        sets = [tuple(name.split("/")) for name in config.sets]
        missing = [f"{s}/{f}" for s, f in sets if (s, f) not in available]
        if missing:
            raise ValueError(f"{run} did not evaluate {missing}; it has {available}")
    default = ResidualTestConfig.model_fields["max_positives"].default
    if config.residual_test.max_positives != default and all(f != "stream" for _, f in sets):
        raise ValueError("max_positives applies to stream validation events; none selected")
    stream_run = "event_source" in experiment
    if stream_run:
        targets, records, seed = _stream_run_inputs(experiment, sets)
    else:
        targets, seed = _training_run_targets(run, experiment, sets)
        records = None
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in config.checkpoints:
        for split, fidelity in sets:
            name = f"{fidelity}_{split}_{checkpoint}"
            path = run / (
                f"stream_{split}_{checkpoint}.npz" if fidelity == "stream" else f"{name}.npz"
            )
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; was checkpoint {checkpoint} evaluated?")
            with np.load(path, allow_pickle=False) as saved:
                labels, logits = saved["labels"], saved["logits"]
            if fidelity == "stream":
                import torch

                from core.event_stream import event_labels

                _same_labels(labels, event_labels(torch.as_tensor(records)).numpy(), name)
                features = records[:, :-1].astype(np.float64)
                metrics, arrays = stream_residual(
                    features, labels, logits, config.residual_test, seed=seed
                )
            else:
                target = targets[split, fidelity]
                _same_labels(labels, np.asarray(target.labels), name)
                metrics, arrays = design_point_residual(
                    target, labels, logits, config.residual_test, seed=seed
                )
            rows.append(
                dict(checkpoint=checkpoint, split=split, fidelity=fidelity, residual_test=metrics)
            )
            np.savez_compressed(output / f"{name}_residual.npz", **arrays)
    report = dict(
        run=str(run),
        resolved_config=config.model_dump(mode="json"),
        seed=seed,
        rows=rows,
    )
    (output / "residual_test.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return output
