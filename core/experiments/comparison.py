"""Paired experiments with explicit variants, shared targets and seed schedules."""

import json
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from core.experiments.datasets import materialize
from core.experiments.evaluation import build_manifest
from core.experiments.runner import run, validate_experiment
from core.experiments.storage import RunStore, write_json
from schemas.experiments import ComparisonSpec, ExperimentSpec, load_experiment


def merge(base, overrides):
    result = deepcopy(base)
    for k, v in overrides.items():
        if (
            isinstance(v, dict)
            and isinstance(result.get(k), dict)
            and not ("kind" in v and v["kind"] != result[k].get("kind"))
        ):
            result[k] = merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def differences(a, b, prefix=""):
    result = {}
    for key in sorted(set(a) | set(b)):
        path = f"{prefix}.{key}" if prefix else key
        left, right = a.get(key), b.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            result.update(differences(left, right, path))
        elif left != right:
            result[path] = {"before": left, "after": right}
    return result


def seeded(spec, seed):
    result = spec.model_copy(deep=True)
    p = result.pipeline
    if p.kind == "count_gp":
        p.backend.seed = seed
    else:
        if p.training:
            if p.training.backend == "neural":
                old = p.training.seed
                stage_seeds = [s.training.seed for s in p.stages or []]
                p.training.seed = seed
                for stage, stage_seed in zip(p.stages or [], stage_seeds, strict=True):
                    stage.training.seed = stage_seed + seed - old
            else:
                p.model.architecture.seed = seed
        if p.spatial_regression:
            p.spatial_regression.seed = seed
    return ExperimentSpec.model_validate(result.model_dump())


def selected_rows(run_path):
    records = json.loads((Path(run_path) / "metrics.json").read_text())
    has_gp = any(r["model"] in {"mfgp", "count_gp"} for r in records)
    if has_gp:
        return [r for r in records if r["model"] in {"mfgp", "count_gp"}]
    return [r for r in records if r["name"].startswith(("event_best_", "event_source_"))]


def summarize(children, output):
    """Separate incompatible targets; never pool discrete and continuous NLL."""
    output = Path(output)
    rows = []
    for child in children:
        for r in selected_rows(child["path"]):
            rows.append(
                {"variant": child["variant"], "seed": child["seed"], "run": child["path"], **r}
            )
    write_json(output / "results.json", rows)
    scalar_metrics = (
        "mean_bias",
        "mae",
        "rmse",
        "pearson_r",
        "spearman_r",
        "crps",
        "weighted_interval_score",
        "average_precision",
        "bernoulli_log_loss",
    )
    grouped = {}
    for r in rows:
        key = (r["partition"], r["target_id"])
        grouped.setdefault(key, []).append(r)
    aggregates, paired = [], []
    baseline_name = children[0]["variant"]
    for (partition, target_id), entries in grouped.items():
        for variant in dict.fromkeys(r["variant"] for r in entries):
            values = [r for r in entries if r["variant"] == variant]
            metrics = {}
            for key in scalar_metrics:
                observed = [v["metrics"][key] for v in values if v["metrics"].get(key) is not None]
                if observed:
                    metrics[key] = {
                        "mean": float(np.mean(observed)),
                        "std": float(np.std(observed, ddof=1)) if len(observed) > 1 else None,
                        "n": len(observed),
                    }
            aggregates.append(
                dict(partition=partition, target_id=target_id, variant=variant, metrics=metrics)
            )
        for r in entries:
            if r["variant"] == baseline_name:
                continue
            base = next(
                (b for b in entries if b["variant"] == baseline_name and b["seed"] == r["seed"]),
                None,
            )
            if base:
                paired.append(
                    dict(
                        partition=partition,
                        target_id=target_id,
                        variant=r["variant"],
                        seed=r["seed"],
                        baseline=baseline_name,
                        delta={
                            k: r["metrics"][k] - base["metrics"][k]
                            for k in scalar_metrics
                            if r["metrics"].get(k) is not None
                            and base["metrics"].get(k) is not None
                        },
                    )
                )
    write_json(
        output / "summary.json",
        {
            "groups": aggregates,
            "paired_differences": paired,
            "grouping": "Matching partition, target identities, counts and weights",
            "nll": "Not pooled; inspect each record with its measure",
            "selection": "Descriptive comparisons only; no winner selected",
        },
    )
    import html

    table = "<table><tr><th>Variant</th><th>Seed</th><th>Partition</th><th>Target group</th>"
    table += "".join(f"<th>{k}</th>" for k in scalar_metrics) + "</tr>"
    for r in rows:
        cells = [r["variant"], r["seed"], r["partition"], r["target_id"][:12]] + [
            r["metrics"].get(k, "—") for k in scalar_metrics
        ]
        table += "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>"
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Controlled comparison</h1>'
        "<p>Compare only matching target groups. No test-based model selection.</p>"
        + table
        + "</table>"
    )
    return aggregates


def compare(path):
    path = Path(path).resolve()
    settings = ComparisonSpec.model_validate(yaml.safe_load(path.read_text()))
    baseline_path = (
        settings.baseline if settings.baseline.is_absolute() else path.parent / settings.baseline
    )
    base = load_experiment(baseline_path)
    variants = []
    for variant in settings.variants:
        value = merge(base.model_dump(mode="json"), variant.overrides)
        if value["pipeline"].get("initial_checkpoint"):
            checkpoint = Path(value["pipeline"]["initial_checkpoint"])
            value["pipeline"]["initial_checkpoint"] = str(
                checkpoint if checkpoint.is_absolute() else (path.parent / checkpoint).resolve()
            )
        candidate = validate_experiment(ExperimentSpec.model_validate(value))
        if (
            candidate.dataset != base.dataset
            or candidate.evaluation != base.evaluation
            or candidate.store != base.store
        ):
            raise ValueError(
                "Controlled variants must share dataset, evaluation protocol and store"
            )
        variants.append((variant.name, candidate))
    store = RunStore(base.store)
    out = store.create(
        {"name": settings.name, "comparison": settings.model_dump(mode="json")}, kind="comparison"
    )
    store.transition(out, "running")
    children = []
    try:
        snapshot = materialize(base.dataset, base.store)
        protocol = build_manifest(snapshot, base.evaluation, out / "evaluation")
        targets = {k: v["target_id"] for k, v in protocol["partitions"].items()}
        write_json(
            out / "variants.json",
            {
                name: differences(base.model_dump(mode="json"), cfg.model_dump(mode="json"))
                for name, cfg in variants
            },
        )
        for seed in settings.seeds:
            for name, cfg in variants:
                concrete = seeded(cfg, seed)
                concrete.name = f"{settings.name}/{name}/seed{seed}"
                concrete.parents = list(dict.fromkeys([*concrete.parents, out.name]))
                child = run(concrete, snapshot=snapshot, expected_targets=targets)
                children.append({"variant": name, "seed": seed, "path": str(child)})
                write_json(out / "children.json", children)
        summarize(children, out)
        store.transition(out, "completed")
    except Exception as exc:
        (out / "failure.txt").write_text(traceback.format_exc())
        store.transition(out, "failed", error=str(exc))
        raise
    return out


def compare_saved(store, identifiers):
    store = RunStore(store)
    records = [store.inspect(i) for i in identifiers]
    if any(r["status"] != "completed" or r["kind"] != "experiment" for r in records):
        raise ValueError("Saved comparisons require completed native experiment runs")
    out = store.create({"name": "saved-comparison"}, parents=identifiers, kind="comparison")
    store.transition(out, "running")
    try:
        children = [{"variant": r["id"], "seed": "saved", "path": r["path"]} for r in records]
        write_json(out / "children.json", children)
        summarize(children, out)
        store.transition(out, "completed")
    except Exception as exc:
        store.transition(out, "failed", error=str(exc))
        raise
    return out
