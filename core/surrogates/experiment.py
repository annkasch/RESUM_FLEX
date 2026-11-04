"""One prepared-data experiment workflow, independent of notebooks/scripts."""

import json

import numpy as np

from core.compute import ComputeLog
from core.gp_subsample import subsample_voxels
from core.surrogate_cnp import split_context_target
from core.surrogates.base import Episode
from core.surrogates.checkpoints import load_surrogate
from core.surrogates.evaluation import evaluate_surrogate, evaluation_threads
from core.surrogates.models import build_surrogate
from schemas.surrogates import SurrogateRunConfig


def prepare_and_open(config):
    """(provider, preparation record) for a run config; rebuilds data from data_config first."""
    from core.surrogates.datasets import load_provider, prepare_data

    record = None
    if config.data_config is not None:
        record = prepare_data(config.data_provider, config.data_config, config.data_directory)
    return load_provider(config.data_provider, config.data_directory), record


def run_experiment(config: SurrogateRunConfig):
    """Fit on LF train; optionally select on LF validation; report available splits.

    Prepared batches are already normalized. Explicitly requested test files
    are opened only after fitting and checkpoint selection have completed.
    Output must be empty, preventing accidental overwriting of prior runs. The compute of
    every stage is written to ``compute.json`` (core.compute).
    """
    config = SurrogateRunConfig.model_validate(config)
    output = config.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_directory; {output} is not empty")
    compute = ComputeLog()
    with compute.stage("data_preparation"):
        provider, data_check = prepare_and_open(config)

    def load_split(split, fidelity):
        return provider.load(config.partitions[split], fidelity)

    batches, provenance = {}, {}
    partitions = [("train", "lf")]
    if config.hf_validation:
        partitions.append(("validation", "hf"))
    if config.lf_validation:
        partitions.insert(1, ("validation", "lf"))
    with compute.stage("data_loading"):
        for split, fidelity in partitions:
            batches[split, fidelity], provenance[f"{split}/{fidelity}"] = load_split(
                split, fidelity
            )
    episodes = {
        key: Episode(
            *split_context_target(
                batch,
                config.validation_context_events,
                seed=config.validation_seed,
                n_pool=config.validation_context_pool,
            )
        )
        for key, batch in batches.items()
    }
    train = batches["train", "lf"]
    selection_episode = episodes.get(("validation", config.selection_fidelity))
    selection_voxels = None
    if selection_episode is not None:
        n = selection_episode.target.batch_size
        index = subsample_voxels(n, config.selection_max_voxels, config.validation_seed, 0)
        selection_voxels = dict(available=n, used=len(index))
        if len(index) < n:
            selection_episode = Episode(
                selection_episode.context.take(index), selection_episode.target.take(index)
            )
    dt = None if train.theta is None else train.theta.shape[-1]
    dp = None if train.phi is None else train.phi.shape[-1]
    seed = (
        config.training.seed
        if config.training.backend == "neural"
        else config.model.architecture.seed
    )
    model = (
        load_surrogate(config.initial_checkpoint)
        if config.initial_checkpoint
        else build_surrogate(config.model, dt, dp, seed=seed)
    )
    if model.config != config.model or (model.dim_theta, model.dim_phi) != (dt, dp):
        raise ValueError("Initial checkpoint architecture/features do not match experiment")
    if not config.initial_checkpoint and hasattr(model, "configure_normalization"):
        # A field combines inputs in physical units; the map is stored with its weights.
        model.configure_normalization(provider.normalization())
    metadata = dict(
        resolved_config=config.model_dump(mode="json"),
        data=provenance,
        context_conditioning=model.uses_context,
        data_config=data_check,
        test_data="Not loaded",
        selection_split=(
            f"validation/{config.selection_fidelity}" if selection_episode is not None else None
        ),
        selection_voxels=selection_voxels,
        evaluation_threads=config.evaluation_threads,
        checkpoint_policy=(
            f"best {config.selection_fidelity.upper()} validation"
            if selection_episode is not None
            else "final training step"
        ),
        training_pr="In-sample",
        score_weights="Uniform natural events; no resampling during evaluation",
    )
    # Inference and the GP stage check that new data carry the same normalization.
    metadata["normalization.json"] = provider.normalization()
    metadata["data_provider"] = config.data_provider
    if hasattr(model, "module"):
        metadata["parameter_count"] = sum(p.numel() for p in model.module.parameters())
    model.metadata = metadata
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    with evaluation_threads(config.evaluation_threads), compute.stage("training"):
        if config.stages:
            from core.surrogates.stages import fit_stages

            result = fit_stages(
                model,
                train,
                config.stages,
                validation=selection_episode,
                checkpoints=output / "checkpoints",
            )
        else:
            result = model.fit(
                train,
                config.training,
                validation=selection_episode,
                selection=config.selection,
                checkpoints=output / "checkpoints",
            )
    (output / "history.json").write_text(json.dumps(result.history, indent=2, allow_nan=False))
    if getattr(config.model, "last_layer", None) is not None:
        with compute.stage("last_layer_fit"):
            _fit_last_layers(output / "checkpoints", train, config)
    (output / "sampling_audit.json").write_text(json.dumps(result.sampling_audit, indent=2))
    with compute.stage("evaluation"):
        _evaluate(config, output, train, batches, episodes, load_split)
    if config.mfgp is not None:
        from core.surrogates.mfgp_stage import run_mfgp_stage

        checkpoint = output / "checkpoints" / config.mfgp_checkpoint
        with compute.stage("mfgp"):
            run_mfgp_stage(config, load_surrogate(checkpoint), checkpoint_path=checkpoint)
    compute.write(output)
    return output


def _evaluate(config, output, train, batches, episodes, load_split):
    """Optional unblinded test, metrics of every checkpoint on every split, plots."""
    # Selection is complete. Never pass test episodes into either training stage.
    test_provenance = {}
    if config.test_fidelities and not config.unblind_test:
        print("Test set blinded (set unblind_test: true to evaluate it)", flush=True)
        (output / "test_evaluation.json").write_text(
            json.dumps({"status": "blinded", "fidelities": config.test_fidelities}, indent=2)
        )
    for fidelity in config.test_fidelities if config.unblind_test else []:
        batch, test_provenance[fidelity] = load_split("test", fidelity)
        batches["test", fidelity] = batch
        if (
            batch.theta is not None
            and train.theta is not None
            and any(np.array_equal(row, other) for row in batch.theta for other in train.theta)
        ):
            raise ValueError("Test voxel coordinates overlap training inputs")
        episodes["test", fidelity] = Episode(
            *split_context_target(
                batch,
                config.validation_context_events,
                seed=config.validation_seed,
                n_pool=config.validation_context_pool,
            )
        )
    if test_provenance:
        (output / "test_evaluation.json").write_text(
            json.dumps(
                {
                    "status": "unblinded",
                    "data": test_provenance,
                    "used_for_selection": False,
                    "context_events": config.validation_context_events,
                    "context_pool": config.validation_context_pool,
                    "context_seed": config.validation_seed,
                },
                indent=2,
            )
        )
    from core.surrogates.readout import context_sweep, readout_metrics

    # Constant-rate reference for the log-loss skill, from the training partition only.
    prevalence = float(train.labels[train.event_mask].mean())
    prevalence = prevalence if 0 < prevalence < 1 else None
    metrics, sweep = [], []
    checkpoints = ["best", "final"]
    if (output / "checkpoints/pretraining/model.json").exists():
        checkpoints.insert(0, "pretraining")
    checkpoints.extend(s.name for s in config.stages or [] if s.name not in checkpoints)
    for checkpoint in checkpoints:
        saved = load_surrogate(output / "checkpoints" / checkpoint)
        for (split, fidelity), episode in episodes.items():
            with evaluation_threads(config.evaluation_threads):
                summary, arrays = evaluate_surrogate(
                    saved, episode.target, context=episode.context, reference_rate=prevalence
                )
                # Training metrics are in-sample; the read-out covers held-out splits only.
                if split != "train":
                    summary.update(
                        readout_metrics(episode.target, arrays, seed=config.validation_seed)
                    )
                    if getattr(saved, "last_layer", None) is not None:
                        summary["voxel_predictive"] = _voxel_predictive(
                            saved, episode, output / f"{fidelity}_{split}_{checkpoint}_predictive"
                        )
                    if saved.uses_context:
                        sweep.extend(
                            dict(checkpoint=checkpoint, split=split, fidelity=fidelity, **row)
                            for row in context_sweep(
                                saved,
                                batches[split, fidelity],
                                config,
                                reference_rate=prevalence,
                            )
                        )
            metrics.append(
                dict(
                    checkpoint=checkpoint,
                    step=saved.metadata["step"],
                    split=split,
                    fidelity=fidelity,
                    **summary,
                )
            )
            np.savez_compressed(output / f"{fidelity}_{split}_{checkpoint}.npz", **arrays)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    if sweep:
        (output / "context_sweep.json").write_text(json.dumps(sweep, indent=2, allow_nan=False))
    from viz.surrogate import plot_surrogate_run

    plot_surrogate_run(output)
    if "pretraining" in checkpoints:
        from viz.surrogate import plot_fine_tuning_comparison

        plot_fine_tuning_comparison(output)


def _fit_last_layers(directory, train, config):
    """Fit the Laplace last layer of every saved checkpoint on the training events.

    Context models condition on the same context split as evaluation; only the remaining
    target events enter the Hessian.
    """
    from core.surrogates.last_layer import fit_last_layer

    for manifest in sorted(directory.rglob("model.json")):
        model = load_surrogate(manifest.parent)
        context, target = None, train
        if model.uses_context:
            context, target = split_context_target(
                train,
                config.validation_context_events,
                seed=config.validation_seed,
                n_pool=config.validation_context_pool,
            )
        fit_last_layer(model, target, context)
        model.save(manifest.parent)


def _voxel_predictive(model, episode, stem):
    """Count-level predictive scores of a Bayesian-last-layer model on one episode.

    Writes ``<stem>.npz`` (per-voxel predictive arrays) and a coverage plot.
    """
    from core.gp_metrics import count_predictive_scores
    from core.surrogates.last_layer import voxel_probability_draws

    target = episode.target
    draws = voxel_probability_draws(model, target, episode.context)  # [S, B]
    hits = np.where(target.event_mask, target.labels, 0).sum(axis=1)
    scores, arrays = count_predictive_scores(
        hits, target.events_per_voxel, draws, seed=model.config.last_layer.seed
    )
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    from viz.dispatch import plot_coverage_test

    intervals = {k: (arrays[f"lower_{k}"], arrays[f"upper_{k}"]) for k in (1, 2, 3)}
    high = max(float(arrays["observed"].max()), float(intervals[3][1].max()))
    plot_coverage_test(
        arrays["observed"],
        arrays["mean"],
        arrays["sigma"],
        out_path=stem.parent / f"{stem.name}_coverage.png",
        title=f"{stem.name}: Bayesian last layer, observation-predictive coverage",
        xlabel="Voxel index",
        predicted_label="Predictive mean",
        raw_label="Observed fraction",
        ylim=(0, high * 1.05),
        intervals=intervals,
        show_predicted=False,
    )
    return scores
