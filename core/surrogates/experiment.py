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
from schemas.data_models import concatenate_batches
from schemas.surrogates import SurrogateRunConfig, uses_context

# experiment.json "evaluation_targets": context models score the events after the context
# pool; context-free models score every event. Runs without the key used the split for all.
AFTER_CONTEXT_POOL, ALL_EVENTS = "after_context_pool", "all_events"


def prepare_and_open(config):
    """(provider, preparation record) for a run config; rebuilds data from data_config first."""
    from core.surrogates.datasets import load_provider, prepare_data

    record = None
    if config.data_config is not None:
        record = prepare_data(config.data_provider, config.data_config, config.data_directory)
    return load_provider(config.data_provider, config.data_directory), record


def evaluation_episode(batch, *, context, n_context, seed, n_pool):
    """Evaluation episode of one batch: the seeded context/target split for a context
    model, all events as targets (no context) otherwise."""
    if not context:
        return Episode(None, batch)
    return Episode(*split_context_target(batch, n_context, seed=seed, n_pool=n_pool))


def run_episode(batch, config):
    """Evaluation episode of one batch for a run config's event model."""
    return evaluation_episode(
        batch,
        context=uses_context(config.model.kind),
        n_context=config.validation_context_events,
        seed=config.validation_seed,
        n_pool=config.validation_context_pool,
    )


def run_experiment(config: SurrogateRunConfig):
    """Fit on LF train (or ``train_sets``); optionally select on validation; report splits.

    Prepared batches are already normalized. Explicitly requested test files
    are opened only after fitting and checkpoint selection have completed.
    Output must be empty, preventing accidental overwriting of prior runs. The compute of
    every stage is written to ``compute.json`` (core.compute).
    """
    config = SurrogateRunConfig.model_validate(config)
    output = config.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_directory; {output} is not empty")
    if config.residual_test is not None:
        from core.surrogates.residual import require_lightgbm

        require_lightgbm()
    compute = ComputeLog()
    with compute.stage("data_preparation"):
        provider, data_check = prepare_and_open(config)

    def load_split(split, fidelity):
        return provider.load(config.partitions[split], fidelity)

    batches, provenance = {}, {}
    label = config.training_label  # "lf" for the default LF train partition
    partitions = []
    if config.lf_validation:
        partitions.append(("validation", "lf"))
    if config.hf_validation:
        partitions.append(("validation", "hf"))
    with compute.stage("data_loading"):
        loaded = [provider.load(*key) for key in config.training_sets]
        batches["train", label] = (
            loaded[0][0] if len(loaded) == 1 else concatenate_batches([b for b, _ in loaded])
        )
        provenance[f"train/{label}"] = (
            loaded[0][1]
            if config.train_sets is None
            else {"/".join(k): d for k, (_, d) in zip(config.training_sets, loaded, strict=True)}
        )
        for split, fidelity in partitions:
            batches[split, fidelity], provenance[f"{split}/{fidelity}"] = load_split(
                split, fidelity
            )
    episodes = {key: run_episode(batch, config) for key, batch in batches.items()}
    train = batches["train", label]
    selection_episode = episodes.get(("validation", config.selection_fidelity))
    selection_voxels = None
    if selection_episode is not None:
        n = selection_episode.target.batch_size
        index = subsample_voxels(n, config.selection_max_voxels, config.validation_seed, 0)
        selection_voxels = dict(available=n, used=len(index))
        if len(index) < n:
            context = selection_episode.context
            selection_episode = Episode(
                None if context is None else context.take(index),
                selection_episode.target.take(index),
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
        evaluation_targets=AFTER_CONTEXT_POOL if model.uses_context else ALL_EVENTS,
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
        episodes["test", fidelity] = run_episode(batch, config)
    if test_provenance:
        record = {"status": "unblinded", "data": test_provenance, "used_for_selection": False}
        if uses_context(config.model.kind):
            record.update(
                targets=AFTER_CONTEXT_POOL,
                context_events=config.validation_context_events,
                context_pool=config.validation_context_pool,
                context_seed=config.validation_seed,
            )
        else:
            record["targets"] = ALL_EVENTS
        (output / "test_evaluation.json").write_text(json.dumps(record, indent=2))
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
                    readout, readout_arrays = readout_metrics(
                        episode.target,
                        arrays,
                        seed=config.validation_seed,
                        residual=config.residual_test,
                        feature_bins=config.feature_calibration.bins,
                        null_draws=config.within_point_null_draws,
                    )
                    summary.update(readout)
                    arrays.update(readout_arrays)
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
        episode = run_episode(train, config)
        fit_last_layer(model, episode.target, episode.context)
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
