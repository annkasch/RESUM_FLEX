"""Shared plots over model-independent experiment artifacts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_surrogate_run(directory):
    directory = Path(directory)
    records = json.loads((directory / "metrics.json").read_text())
    history = json.loads((directory / "history.json").read_text())

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{suffix}", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, key in zip(
        axes, ["voxel_rate_mae", "average_precision", "bernoulli_log_loss"], strict=True
    ):
        ax.plot([r["step"] for r in history], [r.get(key) for r in history])
        ax.set(xlabel="Training step / tree count", ylabel=key)
        ax.grid(alpha=0.2)
    save(fig, "validation_history")
    for fidelity in ("lf", "hf"):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
        for checkpoint in ("best", "final"):
            with np.load(directory / f"{fidelity}_validation_{checkpoint}.npz") as arrays:
                observed, predicted = arrays["observed"], arrays["predicted"]
            axes[0].plot(predicted, "o-", label=checkpoint)
            axes[1].plot(predicted - observed, "o-", label=checkpoint)
        axes[0].plot(observed, "ko-", label="Observed target fraction")
        axes[0].set(ylabel="Detection fraction", title=f"{fidelity.upper()} validation")
        axes[1].axhline(0, color="black", ls="--")
        axes[1].set(xlabel="Validation voxel index", ylabel="Predicted − observed")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        save(fig, f"{fidelity}_means")
        for logscale in (False, True):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
            for ax, checkpoint in zip(axes, ("best", "final"), strict=True):
                for split in ("train", "validation") if fidelity == "lf" else ("validation",):
                    row = next(
                        r
                        for r in records
                        if r["fidelity"] == fidelity
                        and r["split"] == split
                        and r["checkpoint"] == checkpoint
                    )
                    with np.load(directory / f"{fidelity}_{split}_{checkpoint}.npz") as arrays:
                        ax.step(
                            arrays["recall"],
                            arrays["precision"],
                            where="pre",
                            label=f"{split}, AP={row['average_precision']}",
                        )
                    ax.axhline(row["prevalence"], ls="--", alpha=0.4)
                ax.set(xlabel="Recall", ylabel="Precision", title=checkpoint, xlim=(0, 1))
                ax.set(
                    yscale="log" if logscale else "linear",
                    ylim=(1e-4, 1.05) if logscale else (0, 1.05),
                )
                ax.legend(fontsize=8)
                ax.grid(alpha=0.2)
            save(fig, f"{fidelity}_precision_recall" + ("_log" if logscale else ""))
