"""Reports read prediction artifacts, never train or load a model."""

import html
import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from core.experiments.evaluation import evaluate_records
from core.experiments.storage import RunStore, write_json


def render_report(run, output, plots):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core.precision_recall import precision_recall

    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = evaluate_records(run / "predictions")
    write_json(output / "metrics.json", records)
    pages = []
    for r in records:
        name = r["name"]
        with np.load(run / "predictions" / r["file"]) as a:
            for kind in ("means", "residuals", "coverage", "precision_recall"):
                if kind not in plots:
                    continue
                if kind == "coverage" and "lower_1" not in a:
                    continue
                if kind == "precision_recall" and ("logits" not in a or not a["labels"].sum()):
                    continue
                fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
                if kind == "means":
                    ax.plot(a["mean"], label="Predicted mean")
                    ax.plot(a["observed"], "ko", markersize=3, label="Observed target fraction")
                    for k, color in ((3, "lightcoral"), (2, "gold"), (1, "forestgreen")):
                        if f"lower_{k}" in a:
                            nominal = r["metrics"]["interval_metrics"][str(k)][
                                "nominal_probability"
                            ]
                            ax.fill_between(
                                np.arange(len(a["mean"])),
                                a[f"lower_{k}"],
                                a[f"upper_{k}"],
                                color=color,
                                alpha=0.2,
                                label=f"{100 * nominal:.2f}%",
                            )
                    ax.set(xlabel="Evaluation voxel index", ylabel="Detection fraction")
                elif kind == "residuals":
                    ax.plot(a["mean"] - a["observed"], "o", markersize=3)
                    ax.axhline(0, color="black", lw=0.8)
                    ax.set(xlabel="Evaluation voxel index", ylabel="Predicted − observed")
                elif kind == "coverage":
                    d = r["metrics"]["interval_metrics"]
                    x = np.arange(3)
                    ax.bar(
                        x - 0.18,
                        [d[str(k)]["nominal_probability"] for k in (1, 2, 3)],
                        0.36,
                        label="Nominal probability",
                        color="lightgray",
                    )
                    ax.bar(
                        x + 0.18,
                        [d[str(k)]["coverage"] for k in (1, 2, 3)],
                        0.36,
                        label="Exact-coordinate coverage",
                    )
                    for i, k in enumerate((1, 2, 3)):
                        ax.text(
                            i + 0.18,
                            d[str(k)]["coverage"] + 0.02,
                            f"{d[str(k)]['inside']}/{d[str(k)]['total']}",
                            ha="center",
                        )
                    ax.set(
                        xticks=x,
                        xticklabels=["68.27%", "95.45%", "99.73%"],
                        ylim=(0, 1.15),
                        ylabel="Fraction of voxels",
                    )
                else:
                    curve = precision_recall(a["labels"].ravel(), a["logits"].ravel())
                    ax.step(
                        curve["recall"],
                        curve["precision"],
                        where="pre",
                        label=f"AP={curve['average_precision']:.5g}",
                    )
                    ax.axhline(a["labels"].mean(), ls="--", label="Prevalence")
                    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
                ax.set_title(f"{r['model']} — {r['partition']} — {kind}")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=8)
                filename = f"{name}_{kind}.png"
                fig.savefig(output / filename, dpi=120)
                plt.close(fig)
                pages.append(filename)
    # Projection arrays are themselves saved prediction records. Render copies,
    # preserving the completed run and avoiding GP inference during regeneration.
    from core.projection_selection import PLOTS

    wanted = [p for p in plots if p in PLOTS]
    for backend in (run / "backend", run / "backend/mfgp"):
        for folder in ("projections", "projections_observed_fraction"):
            source = backend / folder
            if not (source / "projections.npz").exists():
                continue
            quantity = (
                "observed_fraction" if folder.endswith("observed_fraction") else "latent_mean"
            )
            names = [PLOTS[p][1] for p in wanted if PLOTS[p][0] == quantity]
            if not names:
                continue
            dest = output / folder
            dest.mkdir()
            for f in ("projections.npz", "metadata.json"):
                shutil.copyfile(source / f, dest / f)
            from viz.spatial_projections import plot_spatial_projections

            plot_spatial_projections(dest)
            for name in names:
                pages.append(f"{folder}/{name}.png")
    columns = [
        "model",
        "partition",
        "checkpoint",
        "quantity",
        "distribution",
        "conditioning",
        "mean_bias",
        "mae",
        "rmse",
        "pearson_r",
        "average_precision",
        "outside_probability_range",
    ]
    rows = [{**r, **r["metrics"]} for r in records]
    table = "<table><tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>"
    table += (
        "".join(
            "<tr>"
            + "".join(f"<td>{html.escape(str(r.get(c, '—')))}</td>" for c in columns)
            + "</tr>"
            for r in rows
        )
        + "</table>"
    )
    body = "".join(
        f'<h3>{html.escape(p)}</h3><img src="{html.escape(p)}" style="max-width:100%">'
        for p in pages
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>RESuM experiment</title>'
        "<h1>RESuM experiment results</h1>" + table + body
    )
    return output


def regenerate(store, identifier):
    store = RunStore(store)
    record = store.inspect(identifier)
    if record["status"] != "completed" or record["kind"] == "imported":
        raise ValueError("Regeneration requires a completed native run with prediction records")
    source = Path(record["path"])
    settings = json.loads((source / "resolved_config.json").read_text())["evaluation"]
    return render_report(
        source, store.root / "reports" / identifier / uuid4().hex, settings["plots"]
    )
