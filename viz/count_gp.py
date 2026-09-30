"""Count-GP validation plots in original detection-fraction units."""

import json
from pathlib import Path

import numpy as np

from viz.dispatch import plot_coverage_test


def plot_count_gp(directory):
    import matplotlib.pyplot as plt

    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text())
    for row in metrics:
        group = row["source_group"]
        with np.load(directory / f"{group}_validation.npz") as a:
            intervals = {k: (a[f"lower_{k}"], a[f"upper_{k}"]) for k in (1, 2, 3)}
            for ext in ("png", "pdf"):
                plot_coverage_test(
                    a["observed"],
                    a["mean"],
                    a["observation_sigma"],
                    intervals=intervals,
                    out_path=directory / f"{group}_coverage.{ext}",
                    title=f"{group.upper()} count-GP — binomial predictive intervals",
                    xlabel="Validation voxel index",
                    predicted_label="Detection probability mean",
                    raw_label="Observed hits / all primaries",
                    interval_label="σ-equivalent",
                    ylabel="Detection fraction",
                    nominal_label="Nominal probability",
                    ylim=(0, 1.05 * max(a["observed"].max(), intervals[3][1].max(), 1e-6)),
                )
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
            axes[0].scatter(a["observed"], a["mean"])
            limit = max(a["observed"].max(), a["mean"].max(), 1e-6) * 1.05
            axes[0].plot([0, limit], [0, limit], "k--")
            axes[0].set(
                xlabel="Observed fraction",
                ylabel="Predicted probability",
                title=f"{group.upper()} validation",
            )
            axes[1].scatter(np.arange(len(a["mean"])), a["mean"] - a["observed"])
            axes[1].axhline(0, color="black", linestyle="--")
            axes[1].set(xlabel="Validation voxel index", ylabel="Predicted − observed")
            for ext in ("png", "pdf"):
                fig.savefig(directory / f"{group}_means.{ext}", dpi=150)
            plt.close(fig)
