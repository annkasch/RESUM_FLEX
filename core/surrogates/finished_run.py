"""Inputs of a finished training or stream run, for stages that re-score its saved arrays.

Used by ``residual_stage`` and ``within_point_stage``: held-out sets from the run's
``metrics.json``, inputs reloaded from its resolved config and checked against what the run
recorded (provider provenance, e.g. file hashes, and the saved labels). Never writes to the run.
"""

import json
from pathlib import Path

import numpy as np

from schemas.surrogates import StreamRunConfig, SurrogateRunConfig


def same_labels(saved, loaded, name):
    if saved.shape != loaded.shape or not np.array_equal(saved, loaded):
        raise ValueError(f"{name}: reloaded labels differ from the run's saved labels")


def check_provenance(recorded, loaded, name):
    if recorded is None:
        raise ValueError(f"{name}: the run recorded no provenance for this set")
    if recorded != loaded:
        keys = sorted(k for k in {*recorded, *loaded} if recorded.get(k) != loaded.get(k))
        raise ValueError(f"{name}: data changed since the run (differs in {keys})")


def held_out_sets(run):
    """(split, fidelity) of every held-out set in the run's metrics.json, in order."""
    rows = json.loads((run / "metrics.json").read_text())
    keys = [(r["split"], r["fidelity"]) for r in rows if r["split"] != "train"]
    return list(dict.fromkeys(keys))


def select_sets(run, names, *, available=None):
    """Requested "split/fidelity" names as (split, fidelity), or all ``available`` sets
    (default: every held-out set of the run) for ``names=None``."""
    evaluated = held_out_sets(run)
    if names is None:
        return evaluated if available is None else available
    sets = [tuple(name.split("/")) for name in names]
    missing = [f"{s}/{f}" for s, f in sets if (s, f) not in evaluated]
    if missing:
        raise ValueError(f"{run} did not evaluate {missing}; it has {evaluated}")
    return sets


def saved_arrays_path(run, split, fidelity, checkpoint):
    """The evaluation arrays (logits, labels) the run saved for one set and checkpoint."""
    name = (
        f"stream_{split}_{checkpoint}"
        if fidelity == "stream"
        else (f"{fidelity}_{split}_{checkpoint}")
    )
    path = run / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; was checkpoint {checkpoint} evaluated?")
    return path


def training_run_targets(run, experiment, sets):
    """Target batches of a design-point training run, as its evaluation built them.

    Reads only the data and evaluation-split settings of the run's record, so runs whose
    configs fail current validation elsewhere (e.g. an inert training.context of a
    context-free model) still load. Runs without ``evaluation_targets`` split every set.
    """
    from core.surrogates.datasets import load_provider
    from core.surrogates.experiment import AFTER_CONTEXT_POOL, ALL_EVENTS, evaluation_episode

    resolved = experiment["resolved_config"]
    fields = SurrogateRunConfig.model_fields

    def setting(name):
        return resolved.get(name, fields[name].get_default(call_default_factory=True))

    mode = experiment.get("evaluation_targets", AFTER_CONTEXT_POOL)
    if mode not in (AFTER_CONTEXT_POOL, ALL_EVENTS):
        raise ValueError(f"Unknown evaluation_targets {mode!r} in {run}/experiment.json")
    n_context, seed = setting("validation_context_events"), setting("validation_seed")
    n_pool = max([n_context, *setting("validation_context_sweep")])
    partitions = setting("partitions")
    provider = load_provider(setting("data_provider"), Path(resolved["data_directory"]))
    test_file = run / "test_evaluation.json"
    tests = json.loads(test_file.read_text()).get("data", {}) if test_file.exists() else {}
    targets = {}
    for split, fidelity in sets:
        name = f"{split}/{fidelity}"
        batch, provenance = provider.load(partitions[split], fidelity)
        recorded = tests.get(fidelity) if split == "test" else experiment["data"].get(name)
        check_provenance(recorded, provenance, name)
        targets[split, fidelity] = evaluation_episode(
            batch,
            context=mode == AFTER_CONTEXT_POOL,
            n_context=n_context,
            seed=seed,
            n_pool=n_pool,
        ).target
    return targets, seed


def is_stream_run(experiment):
    """Stream runs list evaluation sets; design-point training runs name partitions."""
    return "evaluation" in experiment["resolved_config"]


def stream_run_inputs(experiment, sets):
    """Design-point batches and the validation records of a stream run.

    Design-point sets need only the provider, data directory, evaluation sets and seed of
    the run's record, so runs made with older stream configs (e.g. before keys were
    renamed) still load them; stream validation records need the full current config.
    """
    from core.surrogates.datasets import load_provider

    resolved = experiment["resolved_config"]
    seed = resolved.get("evaluation_seed", StreamRunConfig.model_fields["evaluation_seed"].default)
    targets, records = {}, None
    grouped = [key for key in sets if key[1] != "stream"]
    if grouped:
        provider = load_provider(resolved.get("data_provider"), Path(resolved["data_directory"]))
        partitions = {(e["split"], e["fidelity"]): e["partition"] for e in resolved["evaluation"]}
        for split, fidelity in grouped:
            name = f"{split}/{fidelity}"
            targets[split, fidelity], provenance = provider.load(
                partitions[split, fidelity], fidelity
            )
            provenance["partition"] = partitions[split, fidelity]
            check_provenance(
                experiment.get("design_point_evaluation", {}).get(name), provenance, name
            )
    if len(grouped) < len(sets):
        from core.event_stream import open_event_source

        config = StreamRunConfig.model_validate(resolved)
        source = open_event_source(
            config.event_source,
            config.event_store,
            "validation",
            **config.event_source_options,
        )
        records = np.ascontiguousarray(source.load()[: experiment["validation_events"]])
    return targets, records, seed


def run_inputs(run, experiment, sets):
    """(targets, stream records or None, evaluation seed) of a training or stream run."""
    if is_stream_run(experiment):
        return stream_run_inputs(experiment, sets)
    targets, seed = training_run_targets(run, experiment, sets)
    return targets, None, seed
