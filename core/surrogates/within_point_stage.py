"""Within-design-point metrics on a finished run (``python -m core.surrogates within_point``).

For runs trained before the ``within_point_*`` metrics existed. Reads the logits and labels
the run saved for each design-point set and checkpoint (``{fidelity}_{split}_{checkpoint}.npz``),
reloads the target batches from the run's resolved config (provenance and labels must match)
and computes what the read-out reports: ``within_point_log_loss_gain``, ``_skill``,
``within_point_auc``, ``within_point_spread_z``, their bootstrap intervals and
``within_point_gain_coverage``, with the run's own evaluation seed, so the values equal the
read-out of the same run trained with the current toolkit. The run directory is never
written to.
"""

import json
from pathlib import Path

import numpy as np

from core.surrogates.evaluation import within_point_summary, within_point_terms
from core.surrogates.finished_run import (
    held_out_sets,
    run_inputs,
    same_labels,
    saved_arrays_path,
    select_sets,
)
from core.surrogates.readout import within_point_bootstrap, within_point_coverage
from schemas.surrogates import WithinPointRunConfig


def within_point_readout(target, labels, logits, *, seed, null_draws):
    """Within-design-point metrics and arrays of one evaluated set, as in the read-out."""
    terms = within_point_terms(target, labels, logits)
    metrics = dict(
        voxels=int(len(terms["gain"])),
        events=int(target.events_per_voxel.sum()),
        positives=int(terms["hits"].sum()),
        **within_point_summary(terms),
        **within_point_bootstrap(terms, seed=seed),
    )
    coverage, null = within_point_coverage(
        target, logits, terms["gain"], terms["variance"], seed=seed, draws=null_draws
    )
    metrics["within_point_gain_coverage"] = coverage
    arrays = dict(
        within_point_gain=terms["gain"],
        within_point_expected_gain=terms["expected"],
        within_point_gain_variance=terms["variance"],
        within_point_gain_null=null,
    )
    return metrics, arrays


def run_within_point(config: WithinPointRunConfig) -> Path:
    """within_point.json, one arrays file and one plot per (set, checkpoint)."""
    from viz.surrogate import plot_within_point_gain

    output = config.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_directory; {output} is not empty")
    run = config.run
    experiment = json.loads((run / "experiment.json").read_text())
    grouped = [key for key in held_out_sets(run) if key[1] != "stream"]
    if not grouped:
        raise ValueError(f"{run} evaluated no design-point sets")
    sets = select_sets(run, config.sets, available=grouped)
    targets, _, seed = run_inputs(run, experiment, sets)
    steps = {
        (r["checkpoint"], r["split"], r["fidelity"]): r.get("step")
        for r in json.loads((run / "metrics.json").read_text())
    }
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in config.checkpoints:
        for split, fidelity in sets:
            name = f"{fidelity}_{split}_{checkpoint}"
            with np.load(saved_arrays_path(run, split, fidelity, checkpoint)) as saved:
                labels, logits = saved["labels"], saved["logits"]
            target = targets[split, fidelity]
            same_labels(labels, np.asarray(target.labels), name)
            metrics, arrays = within_point_readout(
                target, labels, logits, seed=seed, null_draws=config.within_point_null_draws
            )
            step = steps.get((checkpoint, split, fidelity))
            rows.append(
                dict(checkpoint=checkpoint, step=step, split=split, fidelity=fidelity, **metrics)
            )
            np.savez_compressed(output / f"{name}_within_point.npz", **arrays)
            plot_within_point_gain(
                arrays, output / f"{name}_within_point_gain.png", title=f"{name}"
            )
    report = dict(
        run=str(run),
        resolved_config=config.model_dump(mode="json"),
        seed=seed,
        rows=rows,
    )
    (output / "within_point.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return output
