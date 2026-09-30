"""Physical-coordinate marginal plots with clearly identified point observations."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def plot_mfgp_projections(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    settings = metadata["signature"]["config"]
    labels = [f"{name} [{settings['units']}]" for name in settings["axis_labels"]]
    names = settings["axis_labels"]
    domain = (
        "training convex hull" if settings["domain"] == "training_convex_hull" else "specified box"
    )
    subtitle = f"Uniform-volume averages within {domain}; latent uncertainty, no observation noise"
    with np.load(directory / "projections.npz") as data:
        a = dict(data)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{ext}", dpi=160)
        plt.close(fig)

    colors = {1: "#2ca02c", 2: "#e5b82c", 3: "#d65b5b"}
    probabilities = {1: "68.27%", 2: "95.45%", 3: "99.73%"}
    for overlay in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, layout="constrained")
        for i, ax in enumerate(axes):
            x = a[f"axis_{i}"]
            for k in (3, 2, 1):
                ax.fill_between(
                    x,
                    a[f"lower_{k}_{i}"],
                    a[f"upper_{k}_{i}"],
                    color=colors[k],
                    alpha=0.35,
                    label=probabilities[k],
                )
            ax.plot(x, a[f"mean_{i}"], color="C0", label="Marginalized mean")
            if overlay:
                ax.scatter(
                    a["observed_train_theta"][:, i],
                    a["observed_train"],
                    marker="o",
                    facecolors="none",
                    edgecolors="gray",
                    s=25,
                    label="Individual HF training targets",
                    zorder=5,
                )
                ax.scatter(
                    a["observed_validation_theta"][:, i],
                    a["observed_validation"],
                    marker="x",
                    color="black",
                    s=25,
                    label="Individual HF validation targets",
                    zorder=6,
                )
            ax.set(
                xlabel=labels[i],
                title=f"Average over {', '.join(names[j] for j in range(3) if j != i)}",
            )
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Detection fraction")
        axes[-1].legend(fontsize=8)
        title = "Marginalized HF-equivalent response"
        if overlay:
            title += " — observations are individual voxels, not marginal averages"
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
    fig.suptitle("Plane marginalizations: mean and uncertainty widths\n" + subtitle, fontsize=13)
    save(fig, "planes")

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
        for split, marker in (("train", "o"), ("validation", "D")):
            points = a[f"observed_{split}_theta"]
            ax.scatter(
                points[:, i],
                points[:, j],
                c=a[f"observed_{split}"],
                marker=marker,
                norm=norm,
                cmap="viridis",
                edgecolors="black",
                linewidths=0.7,
                s=40,
                zorder=5,
            )
        ax.set(
            xlabel=labels[i],
            ylabel=labels[j],
            aspect="equal",
            title=f"{names[i]}–{names[j]} (average over {names[3 - i - j]})",
        )
    fig.colorbar(mesh, ax=axes, label="Detection fraction (shared map/point scale)", shrink=0.8)
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            color="black",
            markerfacecolor="white",
            linestyle="None",
            label=f"Individual HF {split} targets",
        )
        for split, marker in (("training", "o"), ("validation", "D"))
    ]
    axes[0].legend(handles=handles, fontsize=8)
    fig.suptitle(
        "Marginalized means with observed target fractions\n"
        "Marker colors are individual voxel observations, not marginalized measurements\n"
        + subtitle,
        fontsize=11,
    )
    save(fig, "planes_observed")
