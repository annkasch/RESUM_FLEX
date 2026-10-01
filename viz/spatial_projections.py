"""Physical-coordinate marginal plots with clearly identified point observations."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def projection_counts(arrays, keep):
    """Validation inclusion: interpolated 1D bands, grid-cell 2D bands.

    Only retained coordinates determine membership in a projected cell. Points
    outside its edges or in cells with no finite bands are reported separately.
    """
    points = arrays["observed_validation_theta"]
    observed = arrays["observed_validation"].ravel()
    eligible = np.isfinite(observed)
    indices = []
    for axis in keep:
        edges = arrays[f"edges_{axis}"]
        coordinate = points[:, axis]
        eligible &= np.isfinite(coordinate) & (coordinate >= edges[0]) & (coordinate <= edges[-1])
        index = np.searchsorted(edges, coordinate, side="right") - 1
        indices.append(np.clip(index, 0, len(edges) - 2))
    key = "".join(map(str, keep))
    bounds = {}
    for k in (1, 2, 3):
        lower = arrays[f"lower_{k}_{key}"][tuple(indices)]
        upper = arrays[f"upper_{k}_{key}"][tuple(indices)]
        if len(keep) == 1:
            axis = keep[0]
            edges = arrays[f"edges_{axis}"]
            centers = (edges[:-1] + edges[1:]) / 2
            x = np.r_[edges[0], centers, edges[-1]]
            lo, hi = arrays[f"lower_{k}_{key}"], arrays[f"upper_{k}_{key}"]
            lower = np.interp(points[:, axis], x, np.r_[lo[0], lo, lo[-1]])
            upper = np.interp(points[:, axis], x, np.r_[hi[0], hi, hi[-1]])
        eligible &= np.isfinite(lower) & np.isfinite(upper)
        bounds[k] = (lower, upper)
    total = int(eligible.sum())
    return dict(
        total_validation=len(observed),
        evaluated=total,
        excluded=int(len(observed) - total),
        bands={
            str(k): dict(
                inside=int((eligible & (observed >= lower) & (observed <= upper)).sum()),
                total=total,
            )
            for k, (lower, upper) in bounds.items()
        },
    )


def plot_projection_counts(ax, counts):
    nominal = np.array([0.68268949, 0.95449974, 0.99730020])
    total = counts["evaluated"]
    measured = [counts["bands"][str(k)]["inside"] / total if total else 0 for k in (1, 2, 3)]
    x = np.arange(3)
    ax.bar(x - 0.18, nominal, width=0.36, color="lightgray", label="Nominal probability")
    bars = ax.bar(x + 0.18, measured, width=0.36, color="teal", label="Validation inclusion")
    for k, bar in enumerate(bars, 1):
        inside = counts["bands"][str(k)]["inside"]
        label = f"{inside}/{total}\n{inside / total:.1%}" if total else "N/A"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set(
        xticks=x,
        xticklabels=["68.27%", "95.45%", "99.73%"],
        ylim=(0, 1.28),
        ylabel="Fraction of validation points",
        title=f"Evaluated: {total}/{counts['total_validation']}; excluded: {counts['excluded']}",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.2)


def plot_spatial_projections(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    settings = metadata["signature"]["config"]
    labels = [f"{name} [{settings['units']}]" for name in settings["axis_labels"]]
    names = settings["axis_labels"]
    observation_group = metadata.get("observation_group", "HF")
    response_label = metadata.get("response_label", "HF-equivalent response")
    domain = (
        "training convex hull" if settings["domain"] == "training_convex_hull" else "specified box"
    )
    predictive = settings.get("quantity") == "observed_fraction"
    observation = metadata.get("observation_model") or {}
    noise_label = observation.get("noise_description", "binomial noise")
    budget_label = (f"N={settings['target_events']}; "
                    if settings.get("target_events") is not None else "")
    subtitle = (
        (
            f"Uniform-volume sampling within {domain}; {budget_label}"
            f"spatial variation + GP uncertainty + {noise_label}"
        )
        if predictive
        else (f"Uniform-volume averages within {domain}; latent uncertainty, no observation noise")
    )
    with np.load(directory / "projections.npz") as data:
        a = dict(data)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{ext}", dpi=160)
        plt.close(fig)

    counts = (
        {
            "".join(map(str, keep)): projection_counts(a, keep)
            for keep in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
        }
        if predictive
        else {}
    )
    if predictive:
        (directory / "coverage_counts.json").write_text(
            json.dumps(
                {
                    "plot_version": 2,
                    "definition": "Validation inclusion: interpolated curves and plane grid cells; "
                    "not calibration for the selected validation population",
                    "exclusions": "Outside projected grid edges or undefined projected interval",
                    "projections": counts,
                },
                indent=2,
            )
        )
    colors = {1: "#2ca02c", 2: "#e5b82c", 3: "#d65b5b"}
    probabilities = {1: "68.27%", 2: "95.45%", 3: "99.73%"}
    for overlay in (False, True):
        if overlay and predictive:
            fig, rows = plt.subplots(
                2, 3, figsize=(15, 8), layout="constrained", gridspec_kw={"height_ratios": [2, 1]}
            )
            axes = rows[0]
            for ax in axes[1:]:
                ax.sharey(axes[0])
            for i in range(3):
                plot_projection_counts(rows[1, i], counts[str(i)])
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, layout="constrained")
        for i, ax in enumerate(axes):
            x = (
                np.r_[a[f"edges_{i}"][0], a[f"axis_{i}"], a[f"edges_{i}"][-1]]
                if predictive
                else a[f"axis_{i}"]
            )

            def displayed(values):
                return np.r_[values[0], values, values[-1]] if predictive else values

            for k in (3, 2, 1):
                ax.fill_between(
                    x,
                    displayed(a[f"lower_{k}_{i}"]),
                    displayed(a[f"upper_{k}_{i}"]),
                    color=colors[k],
                    alpha=0.35,
                    label=probabilities[k],
                )
            ax.plot(
                x,
                displayed(a[f"mean_{i}"]),
                color="C0",
                label="Population mean" if predictive else "Marginalized mean",
            )
            if overlay:
                ax.scatter(
                    a["observed_train_theta"][:, i],
                    a["observed_train"],
                    marker="o",
                    facecolors="none",
                    edgecolors="gray",
                    s=25,
                    label=f"Individual {observation_group} training targets",
                    zorder=5,
                )
                ax.scatter(
                    a["observed_validation_theta"][:, i],
                    a["observed_validation"],
                    marker="o",
                    color="black",
                    s=25,
                    label=f"Individual {observation_group} validation targets",
                    zorder=6,
                )
            ax.set(
                xlabel=labels[i],
                title=(
                    ("Sample " if predictive else "Average over ")
                    + ", ".join(names[j] for j in range(3) if j != i)
                ),
            )
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Detection fraction")
        axes[-1].legend(fontsize=8)
        title = (
            "Observed-fraction predictive distribution"
            if predictive
            else f"Marginalized {response_label}"
        )
        if overlay:
            title += (
                " — unconditional bands; nonzero-selected data need not have nominal coverage"
                if predictive
                else " — observations are individual voxels, not marginal averages"
            )
        fig.suptitle(title + "\n" + subtitle, fontsize=11)
        save(fig, "curves_observed" if overlay else "curves")

    planes = ((0, 1), (0, 2), (1, 2))
    means = np.concatenate([a[f"mean_{i}{j}"].ravel() for i, j in planes])
    values = np.concatenate((means, a["observed_train"], a["observed_validation"]))
    norm = Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    fig, axes = plt.subplots(3, 4, figsize=(17, 12), layout="constrained")
    width_norms = {}
    for k in (1, 2, 3):
        high = max(
            float(np.nanmax(a[f"upper_{k}_{i}{j}"] - a[f"lower_{k}_{i}{j}"])) for i, j in planes
        )
        width_norms[k] = Normalize(vmin=0, vmax=high or 1e-12)
    for row, (i, j) in enumerate(planes):
        for col, ax in enumerate(axes[row]):
            values = (
                a[f"mean_{i}{j}"]
                if col == 0
                else (a[f"upper_{col}_{i}{j}"] - a[f"lower_{col}_{i}{j}"])
            )
            mesh = ax.pcolormesh(
                a[f"edges_{i}"],
                a[f"edges_{j}"],
                values.T,
                cmap="viridis" if col == 0 else "magma",
                norm=norm if col == 0 else width_norms[col],
                shading="flat",
            )
            ax.set(
                xlabel=labels[i],
                ylabel=labels[j],
                aspect="equal",
                title=f"{names[i]}–{names[j]}: "
                + ("mean" if col == 0 else f"{probabilities[col]} interval width"),
            )
            fig.colorbar(mesh, ax=ax, shrink=0.75)
    fig.suptitle(
        (
            "Observed-fraction projections: mean and predictive interval widths\n"
            if predictive
            else "Plane marginalizations: mean and uncertainty widths\n"
        )
        + subtitle,
        fontsize=13,
    )
    save(fig, "planes")

    if predictive:
        fig, rows = plt.subplots(
            2, 3, figsize=(16, 9), layout="constrained", gridspec_kw={"height_ratios": [2, 1]}
        )
        axes = rows[0]
        for ax, (i, j) in zip(rows[1], planes, strict=True):
            plot_projection_counts(ax, counts[f"{i}{j}"])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, (i, j) in zip(axes, planes, strict=True):
        mesh = ax.pcolormesh(
            a[f"edges_{i}"],
            a[f"edges_{j}"],
            a[f"mean_{i}{j}"].T,
            norm=norm,
            cmap="viridis",
            shading="flat",
        )
        for split, marker in (("train", "o"), ("validation", "o")):
            points = a[f"observed_{split}_theta"]
            ax.scatter(
                points[:, i],
                points[:, j],
                c=a[f"observed_{split}"],
                marker=marker,
                norm=norm,
                cmap="viridis",
                edgecolors="white" if split == "train" else "black",
                linewidths=0.7,
                s=40,
                zorder=5,
            )
        ax.set(
            xlabel=labels[i],
            ylabel=labels[j],
            aspect="equal",
            title=(
                f"{names[i]}–{names[j]} "
                + f"({'sample' if predictive else 'average over'} {names[3 - i - j]})"
            ),
        )
    fig.colorbar(mesh, ax=axes, label="Detection fraction (shared map/point scale)", shrink=0.8)
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            color="gray" if split == "training" else "black",
            markerfacecolor="white",
            linestyle="None",
            label=f"Individual {observation_group} {split} targets",
        )
        for split, marker in (("training", "o"), ("validation", "o"))
    ]
    axes[0].legend(handles=handles, fontsize=8)
    fig.suptitle(
        (
            "Population mean with observed target fractions\n"
            "Unconditional predictions; nonzero-selected validation is not a coverage test\n"
            if predictive
            else "Marginalized means with observed target fractions\n"
            "Marker colors are individual voxel observations, not marginalized measurements\n"
        )
        + subtitle,
        fontsize=11,
    )
    save(fig, "planes_observed")
