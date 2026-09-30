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


def plot_mfgp_run(directory):
    """Mean/residual and observation coverage diagnostics on held-out voxels."""
    directory = Path(directory)
    for fid in ("lf", "hf"):
        with np.load(directory / f"{fid}_validation.npz") as a:
            obs, mean, sigma = a["observed"], a["mean"], a["sigma"]
            cnp = a["cnp_mean"]
        x = np.arange(len(obs))
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
        axes[0].errorbar(x, mean, yerr=2 * sigma, fmt="o", label="MFGP ±2σ (observation)")
        axes[0].plot(x, cnp, "x", label="CNP mean")
        axes[0].plot(x, obs, "k.", label="Observed target fraction")
        axes[0].set(title=f"{fid.upper()} held-out voxels", ylabel="Detection fraction")
        axes[0].legend()
        axes[1].plot(x, mean - obs, "o-", label="MFGP residual")
        axes[1].axhline(0, color="black", ls="--")
        axes[1].set(xlabel="Validation voxel index", ylabel="Predicted − observed")
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"{fid}_means.{suffix}", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        # Keep a GP-only view readable when the CNP has a large mean offset.
        gp_fig, gp_ax = plt.subplots(figsize=(10, 4), layout="constrained")
        gp_ax.errorbar(x, mean, yerr=2 * sigma, fmt="o", label="MFGP ±2σ (observation)")
        gp_ax.plot(x, obs, "k.-", label="Observed target fraction")
        gp_ax.set(
            xlabel="Validation voxel index",
            ylabel="Detection fraction",
            title=f"{fid.upper()} MFGP validation",
        )
        gp_ax.legend()
        for suffix in ("png", "pdf"):
            gp_fig.savefig(directory / f"{fid}_gp_means.{suffix}", dpi=150)
        plt.close(gp_fig)
        measured = [(np.abs(mean - obs) <= k * sigma).mean() for k in (1, 2, 3)]
        ax.bar(
            np.arange(3) - 0.18, [0.6827, 0.9545, 0.9973], width=0.36, label="Gaussian reference"
        )
        ax.bar(np.arange(3) + 0.18, measured, width=0.36, label="Measured")
        ax.set(
            xticks=np.arange(3),
            xticklabels=["1σ", "2σ", "3σ"],
            ylim=(0, 1.1),
            ylabel="Fraction of validation voxels",
            title=f"{fid.upper()} MFGP coverage",
        )
        ax.legend()
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"{fid}_coverage.{suffix}", dpi=150)
        plt.close(fig)
    with np.load(directory / "development_map.npz") as a:
        theta, mean = a["theta_physical"], a["mean"]
    if theta.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        points = ax.scatter(*theta.T, c=mean, cmap="viridis")
        ax.set(xlabel="x", ylabel="y", zlabel="z", title="HF-equivalent map at development voxels")
        fig.colorbar(points, ax=ax, label="Predicted detection fraction", shrink=0.7)
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"development_map.{suffix}", dpi=150)
        plt.close(fig)
