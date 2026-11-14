"""Residual learnability test on a finished run (``python -m core.surrogates residual``).

Reads the logits and labels the run saved for each held-out set and checkpoint
(``{fidelity}_{split}_{checkpoint}.npz``, ``stream_validation_{checkpoint}.npz``), reloads
the inputs from the run's resolved config and runs ``core.surrogates.residual`` with the
run's own evaluation seed and feature-calibration bins, so the result equals the read-out
of a run trained with the same ``residual_test:`` settings. One correction plot per set
(``{name}_residual_correction.png``) unless the test was skipped. Reloaded inputs must
match the run: provider provenance (e.g. file hashes) and labels. The run directory is
never written to.
"""

import json
from pathlib import Path

import numpy as np

from core.provenance import code_provenance
from core.surrogates.evaluation import CALIBRATION_BINS
from core.surrogates.finished_run import (
    run_inputs,
    same_labels,
    saved_arrays_path,
    select_sets,
)
from core.surrogates.residual import design_point_residual, require_lightgbm, stream_residual
from schemas.surrogates import ResidualRunConfig, ResidualTestConfig


def run_residual(config: ResidualRunConfig) -> Path:
    """residual_test.json and one arrays file per (set, checkpoint) in output_directory."""
    output = config.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_directory; {output} is not empty")
    require_lightgbm()
    code = code_provenance()
    run = config.run
    experiment = json.loads((run / "experiment.json").read_text())
    sets = select_sets(run, config.sets)
    default = ResidualTestConfig.model_fields["max_positives"].default
    if config.residual_test.max_positives != default and all(f != "stream" for _, f in sets):
        raise ValueError("max_positives applies to stream validation events; none selected")
    targets, records, seed = run_inputs(run, experiment, sets)
    code = code_provenance()  # again: now with the provider run_inputs loaded
    bins = (
        experiment["resolved_config"].get("feature_calibration", {}).get("bins", CALIBRATION_BINS)
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in config.checkpoints:
        for split, fidelity in sets:
            name = f"{fidelity}_{split}_{checkpoint}"
            path = saved_arrays_path(run, split, fidelity, checkpoint)
            with np.load(path, allow_pickle=False) as saved:
                labels, logits = saved["labels"], saved["logits"]
            if fidelity == "stream":
                import torch

                from core.event_stream import event_labels

                same_labels(labels, event_labels(torch.as_tensor(records)).numpy(), name)
                features = records[:, :-1].astype(np.float64)
                metrics, arrays = stream_residual(
                    features, labels, logits, config.residual_test, seed=seed, bins=bins
                )
            else:
                target = targets[split, fidelity]
                same_labels(labels, np.asarray(target.labels), name)
                metrics, arrays = design_point_residual(
                    target, labels, logits, config.residual_test, seed=seed, bins=bins
                )
            rows.append(
                dict(checkpoint=checkpoint, split=split, fidelity=fidelity, residual_test=metrics)
            )
            np.savez_compressed(output / f"{name}_residual.npz", **arrays)
            if "residual_correction_events" in arrays:
                from viz.surrogate import plot_residual_correction

                plot_residual_correction(
                    arrays, output / f"{name}_residual_correction.png", title=name
                )
    report = dict(
        run=str(run),
        resolved_config=config.model_dump(mode="json"),
        seed=seed,
        rows=rows,
        code=code,
    )
    (output / "residual_test.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return output
