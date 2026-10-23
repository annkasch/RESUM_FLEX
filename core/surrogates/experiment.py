"""One prepared-data experiment workflow, independent of notebooks/scripts."""

import json

import numpy as np

from core.gp_subsample import subsample_voxels
from core.surrogate_cnp import split_context_target
from core.surrogates.base import Episode
from core.surrogates.checkpoints import load_surrogate
from core.surrogates.evaluation import evaluate_surrogate, evaluation_threads
from core.surrogates.models import build_surrogate
from schemas.surrogates import SurrogateRunConfig

# Role-based prepared data (LF only) stores these splits under CNP role names.
ROLE_PARTITIONS = {"train": "cnp_train", "validation": "cnp_validation", "test": "cnp_test"}


def role_based(root) -> bool:
    """Prepared data uses the CNP role names (LF only) instead of train/validation/test."""
    from data.optical_pipeline import prepared_partitions

    return "cnp_train" in prepared_partitions(root)


def partition_name(root, split):
    """Prepared partition holding a split, in either the split or the role-based layout."""
    return ROLE_PARTITIONS[split] if role_based(root) else split


def load_split(root, split, fidelity):
    """(batch, provenance) of a split; optical_pipeline is looked up at call time."""
    from data import optical_pipeline

    return optical_pipeline.load_partition(root, partition_name(root, split), fidelity)


def ensure_matching_data(data_config, data_directory):
    """Regenerate prepared data whose stored settings or raw inputs differ from data_config."""
    from data.optical_pipeline import ensure_prepared_optical_data
    from schemas.optical import load_optical_config

    preparation = load_optical_config(data_config)
    if preparation.output_directory != data_directory.resolve():
        raise ValueError(
            f"data_config writes to {preparation.output_directory}, "
            f"but data_directory is {data_directory}"
        )
    # config.json is written only by a full preparation.
    saved = data_directory / "config.json"
    before = saved.stat().st_mtime_ns if saved.exists() else None
    ensure_prepared_optical_data(preparation)
    regenerated = before != saved.stat().st_mtime_ns
    print(f"Prepared data {'regenerated' if regenerated else 'up to date'}: {data_directory}",
          flush=True)
    return dict(path=str(data_config), regenerated=regenerated,
                settings=preparation.model_dump(mode="json"))


def run_experiment(config: SurrogateRunConfig):
    """Fit on LF train; optionally select on LF validation; report available splits.

    Prepared batches are already normalized. Explicitly requested test files
    are opened only after fitting and checkpoint selection have completed.
    Output must be empty, preventing accidental overwriting of prior runs.
    """
    config = SurrogateRunConfig.model_validate(config)
    root, output = config.data_directory, config.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_directory; {output} is not empty")
    data_check = None
    if config.data_config is not None:
        data_check = ensure_matching_data(config.data_config, root)
    if role_based(root) and (
        config.hf_validation or "hf" in config.test_fidelities or config.mfgp is not None
    ):
        raise ValueError(
            "Role-based prepared data is LF only; set hf_validation: false, "
            "no hf in test_fidelities and no mfgp stage"
        )
    batches, provenance = {}, {}
    partitions = [("train", "lf")]
    if config.hf_validation:
        partitions.append(("validation", "hf"))
    if config.lf_validation:
        partitions.insert(1, ("validation", "lf"))
    for split, fidelity in partitions:
        batches[split, fidelity], provenance[f"{split}/{fidelity}"] = load_split(
            root, split, fidelity
        )
    episodes = {
        key: Episode(
            *split_context_target(
                batch, config.validation_context_events, seed=config.validation_seed
            )
        )
        for key, batch in batches.items()
    }
    train = batches["train", "lf"]
    selection_episode, selection_voxels = episodes.get(("validation", "lf")), None
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
    model = (load_surrogate(config.initial_checkpoint) if config.initial_checkpoint
             else build_surrogate(config.model, dt, dp, seed=seed))
    if model.config != config.model or (model.dim_theta, model.dim_phi) != (dt, dp):
        raise ValueError("Initial checkpoint architecture/features do not match experiment")
    metadata = dict(
        resolved_config=config.model_dump(mode="json"),
        data=provenance,
        context_conditioning=model.uses_context,
        data_config=data_check,
        test_data="Not loaded",
        selection_split="validation/lf" if config.lf_validation else None,
        selection_voxels=selection_voxels,
        evaluation_threads=config.evaluation_threads,
        checkpoint_policy="best LF validation" if config.lf_validation else "final training step",
        training_pr="In-sample",
        score_weights="Uniform natural events; no resampling during evaluation",
    )
    for name in ("normalization.json", "manifest_snapshot.json", "metadata.json"):
        if (root / name).exists():
            metadata[name] = json.loads((root / name).read_text())
    if hasattr(model, "module"):
        metadata["parameter_count"] = sum(p.numel() for p in model.module.parameters())
    model.metadata = metadata
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    with evaluation_threads(config.evaluation_threads):
        if config.stages:
            from core.surrogates.stages import fit_stages
            result = fit_stages(model, train, config.stages, validation=selection_episode,
                                checkpoints=output / "checkpoints")
        else:
            result = model.fit(
                train, config.training, validation=selection_episode,
                selection=config.selection, checkpoints=output / "checkpoints",
            )
    (output / "history.json").write_text(json.dumps(result.history, indent=2, allow_nan=False))
    (output / "sampling_audit.json").write_text(json.dumps(result.sampling_audit, indent=2))
    # Selection is complete. Never pass test episodes into either training stage.
    test_provenance = {}
    for fidelity in config.test_fidelities:
        batch, test_provenance[fidelity] = load_split(root, "test", fidelity)
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
            )
        )
    if test_provenance:
        (output / "test_evaluation.json").write_text(
            json.dumps(
                {
                    "data": test_provenance,
                    "used_for_selection": False,
                    "context_events": config.validation_context_events,
                    "context_seed": config.validation_seed,
                },
                indent=2,
            )
        )
    metrics = []
    checkpoints = ["best", "final"]
    if (output / "checkpoints/pretraining/model.json").exists():
        checkpoints.insert(0, "pretraining")
    checkpoints.extend(s.name for s in config.stages or [] if s.name not in checkpoints)
    for checkpoint in checkpoints:
        saved = load_surrogate(output / "checkpoints" / checkpoint)
        for (split, fidelity), episode in episodes.items():
            with evaluation_threads(config.evaluation_threads):
                summary, arrays = evaluate_surrogate(saved, episode.target, context=episode.context)
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
    from viz.surrogate import plot_surrogate_run

    plot_surrogate_run(output)
    if "pretraining" in checkpoints:
        from viz.surrogate import plot_fine_tuning_comparison

        plot_fine_tuning_comparison(output)
    if config.mfgp is not None:
        from core.surrogates.mfgp_stage import run_mfgp_stage

        checkpoint = output / "checkpoints" / config.mfgp_checkpoint
        run_mfgp_stage(config, load_surrogate(checkpoint), checkpoint_path=checkpoint)
    return output
