"""Adapters retain legacy defaults; historical provenance is never invented."""

import shutil
from pathlib import Path

import yaml

from core.experiments.storage import RunStore, tree_hashes, write_json
from schemas.experiments import ExperimentSpec


def adapt_config(path):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if "model" in value:
        from schemas.surrogates import load_surrogate_config

        cfg = load_surrogate_config(path)
        partitions = ["train/lf"]
        if cfg.lf_validation:
            partitions.append("validation/lf")
        if cfg.hf_validation:
            partitions.append("validation/hf")
        partitions.extend(f"test/{f}" for f in cfg.test_fidelities)
        return ExperimentSpec(
            name=path.stem,
            store=path.parent / "outputs/experiments",
            dataset={"prepared": cfg.data_directory},
            pipeline={
                "kind": "event",
                "model": cfg.model,
                "training": cfg.training,
                "stages": cfg.stages,
                "initial_checkpoint": cfg.initial_checkpoint,
                "selection": cfg.selection,
                "spatial_regression": cfg.mfgp,
                "spatial_checkpoint": cfg.mfgp_checkpoint,
            },
            evaluation={
                "context_events": cfg.validation_context_events,
                "seed": cfg.validation_seed,
                "partitions": partitions,
                "plots": ["means", "residuals", "coverage", "precision_recall"]
                + (cfg.mfgp.projections.plots if cfg.mfgp and cfg.mfgp.projections.enabled else []),
            },
        )
    from schemas.count_gp import load_count_gp_config

    cfg = load_count_gp_config(path)
    return ExperimentSpec(
        name=path.stem,
        store=path.parent / "outputs/experiments",
        dataset={"preparation": path, "format": "counts"},
        pipeline={
            "kind": "count_gp",
            "backend": cfg.backend,
            "prediction_draws": cfg.prediction_draws,
            "prediction_seed": cfg.prediction_seed,
            "projections": cfg.projections,
        },
        evaluation={
            "context_events": 0,
            "seed": cfg.prediction_seed,
            "partitions": ["validation/hf"]
            + ([] if cfg.data.split.lf_train_only else ["validation/lf"]),
            "plots": ["means", "residuals", "coverage"]
            + (cfg.projections.plots if cfg.projections.enabled else []),
        },
    )


def import_run(source, store):
    source = Path(source).resolve()
    store = RunStore(store)
    path = store.create({"name": source.name, "tags": ["imported"]}, kind="imported")
    store.transition(path, "running")
    try:
        shutil.copytree(source, path / "legacy")
        write_json(
            path / "provenance.json",
            {
                "source": str(source),
                "source_hashes": tree_hashes(source),
                "git_commit": None,
                "environment": None,
                "evaluation_manifest": None,
                "note": "Imported artifacts; missing historical provenance is unknown.",
            },
        )
        store.transition(path, "completed", comparable=False)
    except Exception as exc:
        store.transition(path, "failed", error=str(exc))
        raise
    return path
