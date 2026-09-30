"""One prepared-data experiment workflow, independent of notebooks/scripts."""

import hashlib
import json

import numpy as np

from core.surrogate_cnp import split_context_target
from core.surrogates.base import Episode
from core.surrogates.checkpoints import load_surrogate
from core.surrogates.evaluation import evaluate_surrogate
from core.surrogates.models import build_surrogate
from schemas.surrogates import SurrogateRunConfig


def run_experiment(config: SurrogateRunConfig):
    """Fit on LF train; optionally select on LF validation; report available splits.

    Prepared batches are already normalized. Test files are never opened.
    Output must be empty, preventing accidental overwriting of prior runs.
    """
    from data.optical_pipeline import load_prepared_partition, prepared_partition_paths
    from data.grouped_batches import batch_groups

    config = SurrogateRunConfig.model_validate(config)
    root, output = config.data_directory, config.output_directory
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output_directory; {output} is not empty")
    batches, provenance = {}, {}
    partitions = [("train", "lf"), ("validation", "hf")]
    if config.lf_validation:
        partitions.insert(1, ("validation", "lf"))
    for split, fidelity in partitions:
        batches[split, fidelity] = load_prepared_partition(root, split, fidelity)
        paths = prepared_partition_paths(root, split, fidelity)
        records = [dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                   for path in paths]
        provenance[f"{split}/{fidelity}"] = records[0] if len(records) == 1 else records
    episodes = {}
    for key, batch in batches.items():
        parts = [split_context_target(b, config.validation_context_events, seed=config.validation_seed)
                 for b in batch_groups(batch)]
        episodes[key] = Episode(*parts[0]) if len(parts) == 1 else Episode(
            [p[0] for p in parts], [p[1] for p in parts])
    train = batches["train", "lf"]
    first = batch_groups(train)[0]
    dt = None if first.theta is None else first.theta.shape[-1]
    dp = None if first.phi is None else first.phi.shape[-1]
    seed = (
        config.training.seed
        if config.training.backend == "neural"
        else config.model.architecture.seed
    )
    model = build_surrogate(config.model, dt, dp, seed=seed)
    metadata = dict(
        resolved_config=config.model_dump(mode="json"),
        data=provenance,
        context_conditioning=model.uses_context,
        test_data="Not loaded",
        selection_split="validation/lf" if config.lf_validation else None,
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
    result = model.fit(
        train,
        config.training,
        validation=episodes.get(("validation", "lf")),
        selection=config.selection,
        checkpoints=output / "checkpoints",
    )
    (output / "history.json").write_text(json.dumps(result.history, indent=2, allow_nan=False))
    (output / "sampling_audit.json").write_text(json.dumps(result.sampling_audit, indent=2))
    metrics = []
    for checkpoint in ("best", "final"):
        saved = load_surrogate(output / "checkpoints" / checkpoint)
        for (split, fidelity), episode in episodes.items():
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
    if config.mfgp is not None:
        from core.surrogates.mfgp_stage import run_mfgp_stage

        run_mfgp_stage(config, load_surrogate(output / "checkpoints/best"))
    return output
