"""Predictive scores of a finished event model on design-point sets (``core.surrogates score``).

A context-free checkpoint with a Bayesian last layer predicts each set anew (own event inputs,
or donors with the provider's ``transfer_events``) and is scored on the sets' outcome counts
with the measures of the binomial GP (``readout.predictive_scores``). For sets the run did not
evaluate, e.g. a GP benchmark set, so event models and GPs are scored alike. Writes into its
own new ``output_directory``; the run's files are not modified::

    score.json                                     rows per (checkpoint, set) and the record
    compute.json                                   stages of this scoring run
    <partition>_<fidelity>_<checkpoint>_prediction.npz   ids, coordinates, mean p, draws, donors
    <partition>_<fidelity>_<checkpoint>_predictive.npz   predictive arrays (count_predictive_scores)
    <partition>_<fidelity>_<checkpoint>_coverage.png     observation-predictive coverage
    <partition>_<fidelity>_<checkpoint>_zscore*.png      z = (observed - mean) / sd
"""

import json
from pathlib import Path

import numpy as np

from core.compute import ComputeLog
from core.surrogates.finished_run import is_stream_run
from core.surrogates.inference import (
    event_inputs_record,
    normalization_record,
    open_for_prediction,
    predict_design_points,
    uncertainty_record,
)
from core.surrogates.readout import plot_predictive_coverage, predictive_scores
from schemas.surrogates import ScoreRunConfig, SurrogateRunConfig


def run_sets(resolved) -> tuple[set[tuple[str, str]], str | None]:
    """(training sets as (partition, fidelity), test partition) of a run's resolved config."""
    if "evaluation" in resolved:  # stream run: trained on an event store, no partitions
        return set(), None
    fields = SurrogateRunConfig.model_fields
    partitions = resolved.get("partitions") or fields["partitions"].get_default(
        call_default_factory=True
    )
    train_sets = resolved.get("train_sets")
    training = (
        {(partitions["train"], "lf")}
        if train_sets is None
        else {(t["partition"], t["fidelity"]) for t in train_sets}
    )
    return training, partitions.get("test")


def run_score(config: ScoreRunConfig) -> Path:
    """Score every (checkpoint, set) of ``config``; returns the output directory."""
    from viz.zscore import plot_zscores

    run, output = config.run, config.output_directory
    if output.exists():
        raise FileExistsError(f"{output} exists; scores are never overwritten")
    experiment = json.loads((run / "experiment.json").read_text())
    resolved = experiment["resolved_config"]
    training, test = run_sets(resolved)
    sets = [tuple(name.split("/")) for name in config.sets]
    trained = [f"{p}/{f}" for p, f in sets if (p, f) in training]
    if trained:
        raise ValueError(f"{run} was trained on {trained}; score held-out sets only")
    blinded = [f"{p}/{f}" for p, f in sets if p == test]
    if blinded and not config.unblind_test:
        raise ValueError(f"{blinded} is the run's test partition; set unblind_test: true")
    data_provider = (
        resolved.get("data_provider") if config.data_provider == "run" else config.data_provider
    )
    data_directory = config.data_directory or Path(resolved["data_directory"])
    compute = ComputeLog()
    output.mkdir(parents=True)
    rows = []
    for checkpoint in config.checkpoints:
        path = run / "checkpoints" / checkpoint
        for partition, fidelity in sets:
            donors = (
                None
                if config.donors is None
                else (partition, fidelity)
                if config.donors == "scored"
                else tuple(config.donors.split("/"))
            )
            model, provider, spec = open_for_prediction(
                path, data_directory, data_provider, events_from=donors
            )
            if getattr(model, "last_layer", None) is None:
                raise ValueError(f"{path} has no Bayesian last layer; nothing to score")
            result = predict_design_points(
                model,
                provider,
                partition,
                fidelity,
                events_from=donors,
                seed=config.seed,
                threads=config.threads,
                compute=compute,
            )
            with compute.stage("scoring"):
                target, provenance = provider.load(partition, fidelity)
                hits = np.where(target.event_mask, target.labels, 0).sum(axis=1)  # [B]
                trials = target.events_per_design_point  # [B] the scored set's own counts
                draws = result["uncertainty"]["probability_draws"].astype(float)  # [S, B]
                mean = result["prediction"].mean  # [B] mean p with the trained weights
                scores, arrays = predictive_scores(
                    hits, trials, draws, seed=model.config.last_layer.seed, plugin=mean
                )
            name = f"{partition}_{fidelity}_{checkpoint}"
            np.savez_compressed(
                output / f"{name}_prediction.npz",
                design_point_ids=np.array(result["ids"]),  # [B]
                coordinates=provider.coordinates(partition, fidelity),  # [B, D] physical
                mean_probability=mean,  # [B]
                donor_index=(
                    np.full(len(mean), -1) if result["donors"] is None else result["donors"]
                ),  # [B] row of the donor set, -1: own events
                **result["uncertainty"],
            )
            np.savez_compressed(output / f"{name}_predictive.npz", **arrays)
            with compute.stage("plots"):
                title = f"{run.name}/{checkpoint}, {partition}/{fidelity}"
                plot_predictive_coverage(
                    arrays,
                    output / f"{name}_coverage.png",
                    f"{title}: observation-predictive coverage (Bayesian last layer)",
                    ylim=config.plot_ylim,
                )
                plot_zscores(
                    (arrays["observed"] - arrays["mean"]) / arrays["sigma"],
                    arrays["mean"],
                    output / f"{name}_zscore",
                    title=f"{title}: z = (observed - predictive mean) / predictive sd",
                    rate_label="Predicted rate",
                )
            rows.append(
                dict(
                    checkpoint=checkpoint,
                    step=model.metadata.get("step"),
                    partition=partition,
                    fidelity=fidelity,
                    events=int(trials.sum()),
                    data=provenance,
                    event_inputs=event_inputs_record(donors, config.seed, result["donors"]),
                    normalization_check=normalization_record(model),
                    uncertainty=uncertainty_record(model, result["uncertainty"]),
                    data_provider=spec,
                )
                | dict(design_points=int(len(hits)), hits=int(hits.sum()))
                | scores
            )
    compute.write(output)
    report = dict(
        run=str(run),
        stream_run=is_stream_run(experiment),
        resolved_config=config.model_dump(mode="json"),
        data_directory=str(data_directory),
        rows=rows,
    )
    # Arrays and plots first: score.json marks a finished scoring run.
    (output / "score.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return output
