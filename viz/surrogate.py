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
        keys = ["bernoulli_log_loss", "average_precision", "spearman_r", "calibration_z"]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), layout="constrained")
        for ax, key in zip(axes, keys, strict=True):
            ax.plot([r["step"] for r in history], [r.get(key) for r in history])
            ax.set(xlabel="Training step / tree count", ylabel=key)
            ax.grid(alpha=0.2)
            boundary = next(
                (r["step"] - r["stage_step"] for r in history if r.get("stage") == "fine_tuning"),
                None,
            )
            if boundary is not None:
                ax.axvline(boundary, color="gray", ls="--", label="Start last-layer fitting")
                ax.legend(fontsize=8)
        axes[3].axhspan(-3, 3, color="gray", alpha=0.15, label="|Z| < 3")
        axes[3].legend(fontsize=8)
        save(fig, "validation_history")
    else:
        fig, ax = plt.subplots(layout="constrained")
        ax.plot([r["step"] for r in history], [r.get("training_loss", np.nan) for r in history])
        ax.set(xlabel="Training step", ylabel="Training loss")
        save(fig, "training_history")
    for fidelity in ("lf", "hf"):
        # Per-voxel predicted vs observed fraction; validation keeps its old file name.
        for split, name in (
            ("validation", f"{fidelity}_means"),
            ("test", f"{fidelity}_test_means"),
        ):
            if not (directory / f"{fidelity}_{split}_best.npz").exists():
                continue
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
            for checkpoint in ("best", "final"):
                with np.load(directory / f"{fidelity}_{split}_{checkpoint}.npz") as arrays:
                    observed, predicted = arrays["observed"], arrays["predicted"]
                axes[0].plot(predicted, "o-", label=checkpoint)
                axes[1].plot(predicted - observed, "o-", label=checkpoint)
            axes[0].plot(observed, "ko-", label="Observed target fraction")
            axes[0].set(ylabel="Detection fraction", title=f"{fidelity.upper()} {split}")
            axes[1].axhline(0, color="black", ls="--")
            axes[1].set(xlabel=f"{split.capitalize()} voxel index", ylabel="Predicted − observed")
            for ax in axes:
                ax.legend()
                ax.grid(alpha=0.2)
            save(fig, name)
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
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=8)
                ax.grid(alpha=0.2)
            save(fig, f"{fidelity}_precision_recall" + ("_log" if logscale else ""))
        plot_calibration(directory, fidelity, save)
    if (directory / "context_sweep.json").exists():
        plot_context_sweep(directory, save)


def _wilson(hits, events, z=1.96):
    """Wilson score interval for a binomial fraction."""
    frac = hits / events
    centre = (frac + z**2 / (2 * events)) / (1 + z**2 / events)
    half = z * np.sqrt(frac * (1 - frac) / events + z**2 / (4 * events**2)) / (1 + z**2 / events)
    return centre - half, centre + half


def plot_calibration(directory, fidelity, save):
    """Observed hit fraction vs mean predicted p in equal-count bins, per held-out split."""
    files = [
        (split, checkpoint, directory / f"{fidelity}_{split}_{checkpoint}.npz")
        for split in ("validation", "test")
        for checkpoint in ("best", "final")
    ]
    files = [f for f in files if f[2].exists()]
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5), layout="constrained")
    for split, checkpoint, path in files:
        with np.load(path) as arrays:
            if "calibration_events" not in arrays:
                continue
            events = arrays["calibration_events"]
            predicted = arrays["calibration_predicted"] / events
            observed = arrays["calibration_hits"] / events
            low, high = _wilson(arrays["calibration_hits"], events)
        ax.errorbar(
            predicted,
            observed,
            yerr=[observed - low, high - observed],
            fmt="o-",
            ms=3,
            capsize=2,
            label=f"{split}, {checkpoint}",
        )
    limits = ax.get_xlim()
    ax.plot(limits, limits, "k--", lw=1, label="calibrated")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Mean predicted p (bin)",
        ylabel="Observed hit fraction (bin)",
        title=f"{fidelity.upper()} calibration",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, f"{fidelity}_calibration")


def plot_context_sweep(directory, save):
    """Read-out metrics against context size on one fixed target set."""
    rows = json.loads((directory / "context_sweep.json").read_text())
    keys = ["bernoulli_log_loss", "average_precision", "spearman_r", "calibration_z"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), layout="constrained")
    groups = sorted({(r["fidelity"], r["split"], r["checkpoint"]) for r in rows})
    for ax, key in zip(axes, keys, strict=True):
        for group in groups:
            selected = [r for r in rows if (r["fidelity"], r["split"], r["checkpoint"]) == group]
            observed = [r for r in selected if r["context_labels"] == "observed"]
            line = ax.plot(
                [r["n_context"] for r in observed],
                [r[key] for r in observed],
                "o-",
                label=", ".join(group),
            )[0]
            for r in selected:
                if r["context_labels"] == "zeroed":
                    ax.plot(r["n_context"], r[key], "x", ms=9, color=line.get_color())
        ax.set(xlabel="Context events", ylabel=key)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, title="x = context labels set to 0", title_fontsize=7)
    save(fig, "context_sweep")


def plot_fine_tuning_comparison(directory):
    """Before/after means and ranking on exactly the same saved real targets."""
    directory = Path(directory)
    records = json.loads((directory / "metrics.json").read_text())
    for split, fidelity in sorted({(r["split"], r["fidelity"]) for r in records}):
        before_path = directory / f"{fidelity}_{split}_pretraining.npz"
        after_path = directory / f"{fidelity}_{split}_best.npz"
        with np.load(before_path) as before, np.load(after_path) as after:
            if not np.array_equal(before["labels"], after["labels"]):
                raise ValueError("Fine-tuning comparison requires identical target events")
            x = np.arange(len(before["observed"]))
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
            axes[0].plot(x, before["observed"], "ko", label="Observed target fraction")
            for arrays, label in ((before, "Before fine-tuning"), (after, "After fine-tuning")):
                axes[0].plot(x, arrays["predicted"], "o-", label=label)
                axes[1].plot(x, arrays["residual"], "o-", label=label)
            axes[0].set(
                title=f"{fidelity.upper()} {split} — last-layer fitting",
                ylabel="Detection fraction",
            )
            axes[1].axhline(0, color="black", ls="--")
            axes[1].set(xlabel=f"{split.capitalize()} voxel index", ylabel="Predicted − observed")
            for ax in axes:
                ax.grid(alpha=0.2)
                ax.legend()
            for suffix in ("png", "pdf"):
                fig.savefig(directory / f"{fidelity}_{split}_fine_tuning_means.{suffix}", dpi=150)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
            for arrays, checkpoint, label in (
                (before, "pretraining", "Before fine-tuning"),
                (after, "best", "After fine-tuning"),
            ):
                metric = next(
                    r
                    for r in records
                    if r["split"] == split
                    and r["fidelity"] == fidelity
                    and r["checkpoint"] == checkpoint
                )
                ap = metric["average_precision"]
                ap_label = "undefined" if ap is None else f"{ap:.4g}"
                ax.step(
                    arrays["recall"],
                    arrays["precision"],
                    where="pre",
                    label=f"{label}, AP={ap_label}",
                )
            ax.axhline(metric["prevalence"], ls="--", color="gray", label="Positive fraction")
            ax.set(
                xlabel="Recall",
                ylabel="Precision",
                xlim=(0, 1),
                ylim=(0, 1.05),
                title=f"{fidelity.upper()} {split} — precision–recall",
            )
            ax.legend()
            ax.grid(alpha=0.2)
            for suffix in ("png", "pdf"):
                fig.savefig(directory / f"{fidelity}_{split}_fine_tuning_pr.{suffix}", dpi=150)
            plt.close(fig)


def plot_mfgp_training_inputs(directory):
    """Plot the exact values of every GP level, in training-row order (original units)."""
    directory = Path(directory)
    levels = json.loads((directory / "levels.json").read_text())
    log_mode = json.loads((directory / "model.json").read_text())["config"][
        "output_transform"
    ] == "log"
    fig, axes = plt.subplots(
        1, len(levels), figsize=(5 * len(levels), 4), sharey=True, layout="constrained",
        squeeze=False,
    )
    with np.load(directory / "training_arrays.npz") as raw, \
            np.load(directory / "fit_arrays.npz") as adjusted:
        for ax, level in zip(axes[0], levels, strict=True):
            k = level["level"]
            values = raw[f"Y_{k}"].ravel()
            label = "model mean" if level["source"] == "model_mean" else "observed fraction"
            ax.scatter(np.arange(len(values)), values, s=10, label=label)
            if log_mode and level["source"] == "observed_fraction":
                ax.scatter(np.arange(len(values)), adjusted[f"Y_{k}"].ravel(), marker="+",
                           s=12, label="smoothed before log")
            ax.set(
                title=f"Level {k}: {level['partition']}/{level['fidelity']} ({len(values)})",
                xlabel="GP training voxel index (saved row order)",
                yscale="log" if log_mode else "linear",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Detection fraction")
    fig.suptitle("GP training levels, lowest to highest fidelity")
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"training_inputs.{suffix}", dpi=150)
    plt.close(fig)
    return directory / "training_inputs.png"


def plot_mfgp_run(directory):
    """Training levels and mean/residual/coverage diagnostics per evaluation set."""
    plot_mfgp_training_inputs(directory)
    directory = Path(directory)
    for path in sorted(directory.glob("evaluation_*.npz")):
        with np.load(path) as a:
            obs, mean, sigma = a["observed"], a["mean"], a["sigma"]
            model = a["model_mean"] if "model_mean" in a else None
            intervals = {k: (a[f"lower_{k}"], a[f"upper_{k}"]) for k in (1, 2, 3)}
            is_log = str(a["output_transform"]) == "log"
            label = str(a["label"])
        name = path.stem
        band_label = "95.45% lognormal interval" if is_log else "MFGP ±2σ (observation)"
        x = np.arange(len(obs))
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
        axes[0].fill_between(x, *intervals[2], alpha=0.2, label=band_label)
        axes[0].plot(x, mean, "o", label="MFGP mean")
        if model is not None:
            axes[0].plot(x, model, "x", label="Event-model mean")
        axes[0].plot(x, obs, "k.", label="Observed fraction")
        axes[0].set(title=f"{label}: held-out voxels", ylabel="Detection fraction")
        axes[0].legend()
        axes[1].plot(x, mean - obs, "o-", label="MFGP residual")
        axes[1].axhline(0, color="black", ls="--")
        axes[1].set(xlabel="Evaluation voxel index", ylabel="Predicted − observed")
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"{name}_means.{suffix}", dpi=150)
        plt.close(fig)
        from viz.dispatch import plot_coverage_test

        low = min(float(obs.min()), float(intervals[3][0].min()))
        high = max(float(obs.max()), float(intervals[3][1].max()))
        padding = max((high - low) * 0.05, 1e-6)
        for suffix in ("png", "pdf"):
            plot_coverage_test(
                obs,
                mean,
                sigma,
                out_path=directory / f"{name}_coverage.{suffix}",
                title=f"{label}: MFGP observation-predictive coverage",
                xlabel="Evaluation voxel index",
                predicted_label="MFGP mean",
                raw_label="Observed fraction",
                ylim=(max(0, low - padding) if is_log else low - padding, high + padding),
                intervals=intervals,
                interval_label="σ in log space" if is_log else "σ",
                show_predicted=False,  # the bands carry the mean; the line hid the points
            )
    with np.load(directory / "development_map.npz") as a:
        theta, mean = a["theta_physical"], a["mean"]
    if theta.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        points = ax.scatter(*theta.T, c=mean, cmap="viridis")
        ax.set(xlabel="x", ylabel="y", zlabel="z", title="Target-level map at development voxels")
        fig.colorbar(points, ax=ax, label="Predicted detection fraction", shrink=0.7)
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"development_map.{suffix}", dpi=150)
        plt.close(fig)
