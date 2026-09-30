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

    has_validation = any("voxel_rate_mae" in r for r in history)
    if has_validation:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for ax, key in zip(
            axes, ["voxel_rate_mae", "average_precision", "bernoulli_log_loss"], strict=True
        ):
            ax.plot([r["step"] for r in history], [r.get(key) for r in history])
            ax.set(xlabel="Training step / tree count", ylabel=key)
            ax.grid(alpha=0.2)
        save(fig, "validation_history")
    else:
        fig, ax = plt.subplots(layout="constrained")
        ax.plot([r["step"] for r in history], [r.get("training_loss", np.nan) for r in history])
        ax.set(xlabel="Training step", ylabel="Training loss")
        save(fig, "training_history")
    for fidelity in ("lf", "hf"):
        if (directory / f"{fidelity}_validation_best.npz").exists():
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
                    if not (directory / f"{fidelity}_{split}_{checkpoint}.npz").exists():
                        continue
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
        if not (directory / f"{fid}_validation.npz").exists():
            continue
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
        from viz.dispatch import plot_coverage_test

        low = min(float(obs.min()), float((mean - 3 * sigma).min()))
        high = max(float(obs.max()), float((mean + 3 * sigma).max()))
        padding = max((high - low) * 0.05, 1e-6)
        for suffix in ("png", "pdf"):
            plot_coverage_test(
                obs,
                mean,
                sigma,
                out_path=directory / f"{fid}_coverage.{suffix}",
                title=f"{fid.upper()} MFGP — observation-predictive coverage",
                xlabel="Validation voxel index",
                predicted_label="MFGP mean",
                raw_label="Observed target fraction",
                ylim=(low - padding, high + padding),
            )
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
