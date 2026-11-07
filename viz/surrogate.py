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
        fig.savefig(directory / f"{name}.png", dpi=150)
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
        for split in ("validation", "test"):
            if not (directory / f"{fidelity}_{split}_best.npz").exists():
                continue
            # z-scores of the selected checkpoint: binomial counting noise only.
            with np.load(directory / f"{fidelity}_{split}_best.npz") as arrays:
                if "voxel_z" in arrays:
                    from viz.zscore import plot_zscores

                    plot_zscores(
                        arrays["voxel_z"],
                        arrays["predicted"],
                        directory / f"{fidelity}_{split}_zscore",
                        title=f"{fidelity.upper()} {split}, best checkpoint: "
                        "z = (hits − Σp) / √Σp(1−p)",
                    )
                if "within_point_gain_null" in arrays:
                    plot_within_point_gain(
                        arrays,
                        directory / f"{fidelity}_{split}_within_point_gain.png",
                        title=f"{fidelity.upper()} {split}, best checkpoint",
                    )
                if "feature_calibration_events" in arrays:
                    plot_feature_calibration(
                        arrays,
                        directory / f"{fidelity}_{split}_feature_calibration.png",
                        title=f"{fidelity.upper()} {split}, best checkpoint",
                    )
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
                yscale="log",
                ylim=(1e-4, 1.05),
            )
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        save(fig, f"{fidelity}_precision_recall_log")
        plot_calibration(directory, fidelity, save)
    if (directory / "context_sweep.json").exists():
        plot_context_sweep(directory, save)


def plot_within_point_gain(arrays, out_path, *, title):
    """Log-loss gain per design point against its null bands (labels drawn from p)."""
    from viz.dispatch import plot_coverage_test

    gain = arrays["within_point_gain"]  # [B]
    expected = arrays["within_point_expected_gain"]  # [B]
    null = arrays["within_point_gain_null"]  # [3, 2, B]: ±1/2/3σ quantiles, lower/upper
    intervals = {k: (null[k - 1, 0], null[k - 1, 1]) for k in (1, 2, 3)}
    low = min(float(null[2, 0].min()), float(gain.min()))
    high = max(float(null[2, 1].max()), float(gain.max()))
    pad = 0.05 * (high - low) or 1e-3
    plot_coverage_test(
        gain,
        expected,
        np.sqrt(arrays["within_point_gain_variance"]),
        out_path=out_path,
        title=f"{title}: within-design-point log-loss gain vs model null",
        xlabel="Design point (sorted by expected gain)",
        predicted_label="expected gain (model)",
        raw_label="observed gain",
        ylabel="Σ loss(p̄) − loss(p) per design point [nats]",
        intervals=intervals,
        interval_label="σ-quantile",
        nominal_label="target",
        ylim=(low - pad, high + pad),
        order="predicted",
    )


def plot_feature_calibration(arrays, out_path, *, title):
    """Observed and predicted hit fraction in equal-count bins of each input feature."""
    names = [str(n) for n in arrays["feature_calibration_names"]]
    events = arrays["feature_calibration_events"]  # [F, bins]
    hits, predicted = arrays["feature_calibration_hits"], arrays["feature_calibration_predicted"]
    variance, value = arrays["feature_calibration_variance"], arrays["feature_calibration_value"]
    columns = min(len(names), 4)
    rows = -(-len(names) // columns)
    fig, axes = plt.subplots(
        2 * rows,
        columns,
        figsize=(4 * columns, 4.5 * rows),
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1] * rows},
        layout="constrained",
    )
    for f, name in enumerate(names):
        top, bottom = (
            axes[2 * (f // columns), f % columns],
            axes[2 * (f // columns) + 1, f % columns],
        )
        observed = hits[f] / events[f]
        lower, upper = _wilson(hits[f], events[f])
        top.errorbar(
            value[f],
            observed,
            yerr=[np.maximum(observed - lower, 0), np.maximum(upper - observed, 0)],
            fmt="o",
            ms=3,
            capsize=2,
            color="black",
            label="observed",
        )
        top.plot(value[f], predicted[f] / events[f], "C0-", label="predicted")
        top.set(title=name, ylabel="Hit fraction (bin)")
        top.grid(alpha=0.2)
        z = (hits[f] - predicted[f]) / np.sqrt(variance[f])
        bottom.axhspan(-2, 2, color="gray", alpha=0.15)
        bottom.plot(value[f], z, "o-", ms=3)
        bottom.axhline(0, color="black", lw=0.8)
        bottom.set(xlabel=f"{name} (bin median)", ylabel="z")
        bottom.grid(alpha=0.2)
    for f in range(len(names), rows * columns):
        axes[2 * (f // columns), f % columns].set_visible(False)
        axes[2 * (f // columns) + 1, f % columns].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{title}: calibration per input feature (equal-count bins)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
            # Wilson bounds can miss the point by rounding error at 0 or all hits.
            yerr=[np.maximum(observed - low, 0), np.maximum(high - observed, 0)],
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
                [np.nan if r[key] is None else r[key] for r in observed],  # undefined: gap
                "o-",
                label=", ".join(group),
            )[0]
            for r in selected:
                if r["context_labels"] == "zeroed" and r[key] is not None:
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
            fig.savefig(directory / f"{fidelity}_{split}_fine_tuning_means.png", dpi=150)
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
            fig.savefig(directory / f"{fidelity}_{split}_fine_tuning_pr.png", dpi=150)
            plt.close(fig)


def plot_mfgp_training_inputs(directory):
    """Plot the exact values of every GP level, in training-row order (original units)."""
    directory = Path(directory)
    levels = json.loads((directory / "levels.json").read_text())
    log_mode = (
        json.loads((directory / "model.json").read_text())["config"]["output_transform"] == "log"
    )
    fig, axes = plt.subplots(
        1,
        len(levels),
        figsize=(5 * len(levels), 4),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    with (
        np.load(directory / "training_arrays.npz") as raw,
        np.load(directory / "fit_arrays.npz") as adjusted,
    ):
        for ax, level in zip(axes[0], levels, strict=True):
            k = level["level"]
            values = raw[f"Y_{k}"].ravel()
            label = "model mean" if level["source"] == "model_mean" else "observed fraction"
            ax.scatter(np.arange(len(values)), values, s=10, label=label)
            if log_mode and level["source"] == "observed_fraction":
                ax.scatter(
                    np.arange(len(values)),
                    adjusted[f"Y_{k}"].ravel(),
                    marker="+",
                    s=12,
                    label="smoothed before log",
                )
            ax.set(
                title=f"Level {k}: {level['partition']}/{level['fidelity']} ({len(values)})",
                xlabel="GP training voxel index (saved row order)",
                yscale="log" if log_mode else "linear",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Detection fraction")
    fig.suptitle("GP training levels, lowest to highest fidelity")
    fig.savefig(directory / "training_inputs.png", dpi=150)
    plt.close(fig)
    return directory / "training_inputs.png"


def plot_mfgp_run(directory, coverage_ylim=None):
    """Training levels and mean/residual/coverage diagnostics per evaluation set.

    coverage_ylim: fixed y range of the coverage plots; None reads ``coverage_ylim`` from the
    run's model.json config, and if that is unset the range follows the data.
    """
    plot_mfgp_training_inputs(directory)
    directory = Path(directory)
    if coverage_ylim is None:
        coverage_ylim = json.loads((directory / "model.json").read_text())["config"].get(
            "coverage_ylim"
        )
    from viz.zscore import gp_zscores, plot_zscores

    pseudocount = json.loads((directory / "model.json").read_text())["config"]["raw_pseudocount"]
    for path in sorted(directory.glob("evaluation_*.npz")):
        with np.load(path) as a:
            log_mode = str(a["output_transform"]) == "log"
            plot_zscores(
                gp_zscores(a, pseudocount),
                a["mean"],
                directory / f"{path.stem}_zscore",
                title=f"{a['label']}: z = (observed − GP mean) / GP σ"
                + (" in log space" if log_mode else ""),
                rate_label="GP predicted rate",
            )
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
        fig.savefig(directory / f"{name}_means.png", dpi=150)
        plt.close(fig)
        from viz.dispatch import plot_coverage_test

        low = min(float(obs.min()), float(intervals[3][0].min()))
        high = max(float(obs.max()), float(intervals[3][1].max()))
        padding = max((high - low) * 0.05, 1e-6)
        plot_coverage_test(
            obs,
            mean,
            sigma,
            out_path=directory / f"{name}_coverage.png",
            title=f"{label}: MFGP observation-predictive coverage",
            xlabel="Evaluation voxel index",
            predicted_label="MFGP mean",
            raw_label="Observed fraction",
            ylim=tuple(coverage_ylim)
            if coverage_ylim
            else (max(0, low - padding) if is_log else low - padding, high + padding),
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
        fig.savefig(directory / "development_map.png", dpi=150)
        plt.close(fig)


def plot_stream_run(directory):
    """Plots of a field trained on events, beside plot_surrogate_run's voxel plots:
    stream log loss history, and calibration and precision-recall on the validation events.
    """
    directory = Path(directory)
    records = json.loads((directory / "metrics.json").read_text())
    history = json.loads((directory / "history.json").read_text())

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=150)
        plt.close(fig)

    reference = next((r["reference_log_loss"] for r in records if r["fidelity"] == "stream"), None)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    steps = [r["step"] for r in history]
    ax.plot(steps, [r["training_loss"] for r in history], label="training batches")
    ax.plot(steps, [r["stream_log_loss"] for r in history], label="validation events")
    if reference is not None:
        ax.axhline(reference, color="gray", ls="--", label="constant training rate")
    ax.set(xlabel="Training step", ylabel="Bernoulli log loss", title="Stream log loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "stream_history")
    plot_calibration(directory, "stream", save)
    if (directory / "stream_validation_best.npz").exists():
        with np.load(directory / "stream_validation_best.npz") as arrays:
            if "feature_calibration_events" in arrays:
                plot_feature_calibration(
                    arrays,
                    directory / "stream_feature_calibration.png",
                    title="Stream validation events, best checkpoint",
                )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, checkpoint in zip(axes, ("best", "final"), strict=True):
        path = directory / f"stream_validation_{checkpoint}.npz"
        if not path.exists():
            continue
        row = next(
            r for r in records if r["fidelity"] == "stream" and r["checkpoint"] == checkpoint
        )
        with np.load(path) as arrays:
            ax.step(
                arrays["recall"],
                arrays["precision"],
                where="pre",
                label=f"validation events, AP={row['average_precision']}",
            )
        ax.axhline(row["prevalence"], ls="--", alpha=0.4)
        ax.set(xlabel="Recall", ylabel="Precision", title=checkpoint, xlim=(0, 1))
        ax.set(
            yscale="log",
            ylim=(1e-4, 1.05),
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    save(fig, "stream_precision_recall_log")
