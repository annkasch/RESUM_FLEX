"""One lifecycle for event surrogates, downstream MFGPs and direct count GPs."""

import json
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.experiments.datasets import materialize
from core.experiments.evaluation import build_manifest, evaluate_records, save_record
from core.experiments.storage import RunStore, provenance, tree_hashes, write_json
from schemas.experiments import ExperimentSpec, load_experiment


def validate_experiment(spec):
    spec = (
        load_experiment(spec)
        if isinstance(spec, (str, Path))
        else ExperimentSpec.model_validate(spec)
    )
    if spec.dataset.prepared and not spec.dataset.prepared.is_dir():
        raise FileNotFoundError(spec.dataset.prepared)
    if spec.dataset.preparation and not spec.dataset.preparation.is_file():
        raise FileNotFoundError(spec.dataset.preparation)
    if spec.pipeline.kind == "event" and spec.pipeline.initial_checkpoint:
        path = spec.pipeline.initial_checkpoint
        from core.surrogates.checkpoints import load_surrogate

        model = load_surrogate(path)
        if model.config != spec.pipeline.model:
            raise ValueError("Checkpoint architecture differs from requested model")
        if spec.pipeline.fit and model.config.kind == "bdt":
            raise ValueError("BDT continuation from checkpoint is unsupported")
    return spec


def projection_settings(spec):
    from schemas.projections import ProjectionConfig

    pipeline = spec.pipeline
    settings = (
        pipeline.spatial_regression.projections.model_copy(deep=True)
        if pipeline.kind == "event" and pipeline.spatial_regression
        else pipeline.projections.model_copy(deep=True)
        if pipeline.kind == "count_gp"
        else ProjectionConfig()
    )
    settings.plots = [
        p for p in spec.evaluation.plots if p.startswith(("projected_", "marginalized_"))
    ]
    settings.enabled = bool(settings.plots)
    return settings


def event_config(spec, data, out):
    from schemas.surrogates import NeuralTraining, SurrogateRunConfig, TreeTraining

    p, e = spec.pipeline, spec.evaluation
    training = p.training or (
        TreeTraining()
        if p.model.kind == "bdt"
        else NeuralTraining(loss="theory-truth" if p.model.kind == "legacy_cnp" else "bernoulli")
    )
    spatial = None if p.spatial_regression is None else p.spatial_regression.model_copy(deep=True)
    if spatial:
        spatial.projections = projection_settings(spec)
    return SurrogateRunConfig(
        model=p.model,
        training=training,
        stages=p.stages,
        initial_checkpoint=p.initial_checkpoint,
        selection=p.selection,
        mfgp=spatial,
        mfgp_checkpoint=p.spatial_checkpoint,
        data_directory=data,
        output_directory=out,
        lf_validation="validation/lf" in e.partitions,
        hf_validation="validation/hf" in e.partitions,
        test_fidelities=[f for f in ("lf", "hf") if f"test/{f}" in e.partitions],
        validation_context_events=e.context_events,
        validation_seed=e.seed,
    )


def _gather(batch, indices):
    from schemas.data_models import StandardBatch

    phi = None if batch.phi is None else np.take_along_axis(batch.phi, indices[:, :, None], axis=1)
    return StandardBatch(
        mode=batch.mode,
        theta=batch.theta,
        phi=phi,
        labels=np.take_along_axis(batch.labels, indices, axis=1),
    )


def record_events(spec, data, output, protocol, checkpoints):
    from core.surrogates.checkpoints import load_surrogate
    from data.optical_pipeline import load_prepared_batch

    for label, checkpoint in checkpoints.items():
        model = load_surrogate(checkpoint)
        for partition, info in protocol["partitions"].items():
            batch = load_prepared_batch(data / "batches" / (partition + ".npz"))
            with np.load(output / "evaluation" / info["file"]) as ids:
                target = _gather(batch, ids["target_indices"])
                context = _gather(batch, ids["context_indices"])
                prediction = model.predict(target, context=context)
                arrays = dict(
                    observed=target.labels.mean(1),
                    mean=prediction.mean,
                    logits=prediction.logits,
                    labels=target.labels,
                    hits=ids["hits"],
                    trials=ids["trials"],
                )
                if "coordinates" in ids:
                    arrays["coordinates"] = ids["coordinates"]
            save_record(
                output / "predictions",
                f"event_{label}_{partition.replace('/', '_')}",
                arrays,
                partition=partition,
                protocol=protocol,
                model=spec.pipeline.model.kind,
                checkpoint=str(checkpoint),
                quantity="event_score_mean",
                conditioning={
                    "uses_context": model.uses_context,
                    "context_events_per_voxel": spec.evaluation.context_events
                    if model.uses_context
                    else 0,
                },
            )


def record_gp(spec, output, protocol, gp, *, kind):
    from math import erf, sqrt

    from scipy.special import expit, logsumexp
    from scipy.stats import binom, lognorm, norm

    normpath = output / "backend/mfgp/normalization.json"
    normalization = json.loads(normpath.read_text()) if normpath.exists() else {}
    for partition, info in protocol["partitions"].items():
        split, fid = partition.split("/")
        with np.load(output / "evaluation" / info["file"]) as ids:
            if "coordinates" not in ids:
                raise ValueError("Spatial GP evaluation requires coordinates")
            x = ids["coordinates"]
            hits, trials = ids["hits"], ids["trials"]
            observed = hits / trials
            if kind == "mfgp":
                if normalization.get("theta"):
                    x = (x - np.asarray(normalization["theta"]["offset"])) / np.asarray(
                        normalization["theta"]["scale"]
                    )
                level = 0 if fid == "lf" else 2
                mean, _ = gp.predict(x, fidelity=level)
                mu, var = gp.predict_transformed(x, fidelity=level)
                sigma = np.sqrt(var)
                draws = np.random.default_rng(spec.evaluation.seed).normal(
                    mu, sigma, size=(4096, len(mu))
                )
                islog = gp.output_transform == "log"
                samples = np.exp(draws) if islog else draws
                median = np.exp(mu) if islog else mu
                logprob = (
                    lognorm.logpdf(observed, s=sigma, scale=np.exp(mu))
                    if islog
                    else norm.logpdf(observed, loc=mu, scale=sigma)
                )
                intervals = {
                    k: gp.predict_interval(x, fidelity=level, n_sigma=k) for k in (1, 2, 3)
                }
                distribution = {
                    "family": "lognormal" if islog else "Gaussian",
                    "measure": "continuous_density",
                    "quantity": "observed_fraction",
                    "noise": "learned_Gaussian_in_fitted_space",
                }
            else:
                cfg = spec.pipeline
                mean, _ = gp.predict(x)
                samples = gp.predict_observations(
                    x, trials, n_draws=cfg.prediction_draws, seed=cfg.prediction_seed
                )
                median = np.median(samples, axis=0)
                intervals = {
                    k: np.quantile(
                        samples,
                        [(1 - erf(k / sqrt(2))) / 2, 1 - (1 - erf(k / sqrt(2))) / 2],
                        axis=0,
                        method="inverted_cdf",
                    )
                    for k in (1, 2, 3)
                }
                mu, var = gp.predict_latent(x)
                prob = expit(
                    np.random.default_rng(cfg.prediction_seed).normal(
                        mu, np.sqrt(var), size=(cfg.prediction_draws, len(x))
                    )
                )
                logprob = logsumexp(binom.logpmf(hits, trials, prob), axis=0) - np.log(len(prob))
                distribution = {
                    "family": "binomial_logistic_GP",
                    "measure": "count_mass",
                    "quantity": "observed_fraction",
                    "noise": "binomial",
                }
            arrays = dict(
                mean=mean,
                observed=observed,
                hits=hits,
                trials=trials,
                coordinates=ids["coordinates"],
                samples=samples,
                median=median,
                log_probability=logprob,
                **{
                    f"{bound}_{k}": intervals[k][i]
                    for k in (1, 2, 3)
                    for i, bound in enumerate(("lower", "upper"))
                },
            )
        save_record(
            output / "predictions",
            f"{kind}_{partition.replace('/', '_')}",
            arrays,
            partition=partition,
            protocol=protocol,
            model=kind,
            checkpoint=str(
                output / ("backend/mfgp/model.pkl" if kind == "mfgp" else "backend/model.pkl")
            ),
            quantity="observed_fraction_mean",
            distribution=distribution,
        )


def count_data(data, protocol, evaluation):
    """Train on full available counts; evaluate only the manifest's target events."""
    data = Path(data)
    trainpath = data / "train.npz"
    if trainpath.exists():
        with np.load(trainpath) as a:
            train = {k: a[k] for k in a.files}
    else:
        chunks = []
        meta = (
            json.loads((data / "metadata.json").read_text())
            if (data / "metadata.json").exists()
            else {}
        )
        norm = (
            json.loads((data / "normalization.json").read_text())
            if (data / "normalization.json").exists()
            else {}
        )
        for fid in ("lf", "hf"):
            path = data / f"batches/train/{fid}.npz"
            if not path.exists():
                continue
            with np.load(path) as a:
                coordinates = a["theta"]
                if norm.get("theta"):
                    coordinates = coordinates * np.asarray(norm["theta"]["scale"]) + np.asarray(
                        norm["theta"]["offset"]
                    )
                files = meta.get("train", {}).get(fid, {}).get("files") or [
                    f"train/{fid}/row{i}" for i in range(len(coordinates))
                ]
                chunks.append(
                    dict(
                        theta=coordinates,
                        hits=a["labels"].sum(1).astype(int),
                        trials=np.full(len(coordinates), a["labels"].shape[1]),
                        files=np.asarray(files),
                        source_group=np.full(len(coordinates), fid),
                    )
                )
        if not chunks:
            raise ValueError("No count-GP training data")
        train = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    valid = []
    for partition, entry in protocol["partitions"].items():
        if not partition.startswith("validation/"):
            continue
        with np.load(Path(evaluation) / entry["file"]) as a:
            valid.append(
                dict(
                    theta=a["coordinates"],
                    hits=a["hits"],
                    trials=a["trials"],
                    files=a["files"],
                    source_group=np.full(len(a["hits"]), partition.split("/")[1]),
                )
            )
    if not valid:
        raise ValueError("Count-GP requires at least one validation partition")
    validation = {k: np.concatenate([c[k] for c in valid]) for k in valid[0]}
    rows = [
        {
            "file": str(f),
            "source_group": str(g),
            "split": split,
            "theta": x.tolist(),
            "hits": int(h),
            "trials": int(n),
        }
        for split, a in [("train", train), ("validation", validation)]
        for f, g, x, h, n in zip(
            a["files"], a["source_group"], a["theta"], a["hits"], a["trials"], strict=True
        )
    ]
    return {"train": train, "validation": validation}, rows, {"evaluation_manifest": protocol["id"]}


def run(spec, *, snapshot=None, expected_targets=None):
    spec = validate_experiment(spec)
    store = RunStore(spec.store)
    out = store.create(spec.model_dump(mode="json"), parents=spec.parents)
    captured = []
    store.transition(out, "running")
    try:
        write_json(out / "provenance.json", provenance())
        write_json(out / "resolved_config.json", spec.model_dump(mode="json"))
        snapshot = Path(snapshot) if snapshot else materialize(spec.dataset, spec.store)
        from core.experiments.storage import verify_dataset

        snapshot_meta = verify_dataset(snapshot)
        write_json(
            out / "dataset.json",
            {
                "snapshot": str(snapshot),
                **snapshot_meta,
                "source": spec.dataset.model_dump(mode="json"),
            },
        )
        data = snapshot / "data"
        protocol = build_manifest(snapshot, spec.evaluation, out / "evaluation")
        targets = {k: v["target_id"] for k, v in protocol["partitions"].items()}
        if expected_targets is not None and targets != expected_targets:
            raise ValueError("Comparison evaluation targets differ")
        p = spec.pipeline
        if p.kind == "event" and p.initial_checkpoint:
            write_json(
                out / "checkpoint_dependency.json",
                {"path": str(p.initial_checkpoint), "hashes": tree_hashes(p.initial_checkpoint)},
            )
            # Normalization is part of the checkpoint input contract when available.
            from core.surrogates.checkpoints import load_surrogate

            source_model = load_surrogate(p.initial_checkpoint)
            norm = source_model.metadata.get("normalization.json")
            if norm is not None and (data / "normalization.json").exists():
                if norm != json.loads((data / "normalization.json").read_text()):
                    raise ValueError("Checkpoint normalization differs from dataset snapshot")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if p.kind == "event":
                cfg = event_config(spec, data, out / "backend")
                if p.fit:
                    from core.surrogates.experiment import run_experiment

                    run_experiment(cfg)
                    checkpoints = {
                        "best": out / "backend/checkpoints/best",
                        "final": out / "backend/checkpoints/final",
                    }
                    for name in ["pretraining"] + [s.name for s in p.stages or []]:
                        path = out / "backend/checkpoints" / name
                        if path.exists():
                            checkpoints[name] = path
                else:
                    cfg.output_directory.mkdir()
                    checkpoints = {"source": p.initial_checkpoint}
                    if cfg.mfgp:
                        from core.surrogates.mfgp_stage import run_mfgp_stage

                        run_mfgp_stage(cfg, source_model, checkpoint_path=p.initial_checkpoint)
                record_events(spec, data, out, protocol, checkpoints)
                if cfg.mfgp:
                    from core.surrogate_mfgp import load_mfgp

                    gp = load_mfgp(out / "backend/mfgp/model.pkl")
                    record_gp(spec, out, protocol, gp, kind="mfgp")
            else:
                from core.binomial_gp import BinomialGP
                from core.count_gp_experiment import run_count_gp

                cfg = SimpleNamespace(
                    backend=p.backend,
                    output_directory=out / "backend",
                    prediction_draws=p.prediction_draws,
                    prediction_seed=p.prediction_seed,
                    projections=projection_settings(spec),
                )
                cfg.model_dump_json = lambda **kw: json.dumps(spec.model_dump(mode="json"), **kw)
                run_count_gp(cfg, prepared=count_data(data, protocol, out / "evaluation"))
                record_gp(
                    spec, out, protocol, BinomialGP.load(out / "backend/model.pkl"), kind="count_gp"
                )
            write_json(out / "metrics.json", evaluate_records(out / "predictions"))
            from core.experiments.reporting import render_report

            render_report(out, out / "report", spec.evaluation.plots)
        if p.kind == "event" and p.initial_checkpoint:
            dependency = json.loads((out / "checkpoint_dependency.json").read_text())
            if tree_hashes(p.initial_checkpoint) != dependency["hashes"]:
                raise ValueError("Source checkpoint changed during the experiment")
        write_json(
            out / "warnings.json",
            [{"category": w.category.__name__, "message": str(w.message)} for w in captured],
        )
        store.transition(
            out, "completed", dataset_id=snapshot_meta["id"], evaluation_id=protocol["id"]
        )
    except Exception as exc:
        write_json(
            out / "warnings.json",
            [{"category": w.category.__name__, "message": str(w.message)} for w in captured],
        )
        (out / "failure.txt").write_text(traceback.format_exc())
        store.transition(out, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    return out
