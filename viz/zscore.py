"""Per-voxel z-scores of observations against predictions, for event models and GPs.

z = (observed - predicted) / expected spread, so every voxel is on the same scale. A model
that is right has mean 0 and spread 1 everywhere, also within bins of the predicted rate.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def gp_zscores(arrays, pseudocount):
    """z [B] of a saved GP evaluation (evaluation_<e>.npz) in the GP's own space.

    The GP's predictive mean and sigma come from the saved ±1σ interval: in log mode it
    is exp(mu ± sigma), and the observation is the smoothed fraction (m + a)/(N + 2a),
    as for the GP targets.
    """
    lower, upper = arrays["lower_1"], arrays["upper_1"]
    observed = arrays["observed"]
    if str(arrays["output_transform"]) == "log":
        n = arrays["target_events"]
        observed = np.log((observed * n + pseudocount) / (n + 2 * pseudocount))
        lower, upper = np.log(lower), np.log(upper)
    mean, sigma = (lower + upper) / 2, (upper - lower) / 2
    return (observed - mean) / sigma


def plot_zscores(z, predicted, out_path, *, title, rate_label="Predicted rate"):
    """Histogram of z against N(0, 1), and z against the predicted rate with ±1/2/3σ bands.

    z, predicted: [B] per voxel. Writes PNG and PDF next to ``out_path`` (no suffix).
    Returns the summary (mean, std, fraction |z| > 2).
    """
    z, predicted = np.asarray(z, dtype=float), np.asarray(predicted, dtype=float)
    if z.shape != predicted.shape or not np.isfinite(z).all():
        raise ValueError("Need finite z and predicted rates of equal shape")
    summary = dict(mean=float(z.mean()), std=float(z.std(ddof=1)) if len(z) > 1 else 0.0,
                   outside_2=float(np.mean(np.abs(z) > 2)), voxels=len(z))
    fig, (ax_hist, ax_rate) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    limit = max(4.0, float(np.ceil(np.abs(z).max())))
    ax_hist.hist(z, bins=np.linspace(-limit, limit, 41), density=True, color="C0", alpha=0.6,
                 label=f"{len(z)} voxels")
    grid = np.linspace(-limit, limit, 400)
    ax_hist.plot(grid, norm.pdf(grid), "k--", label="N(0, 1)")
    ax_hist.set(xlabel="z = (observed − predicted) / σ", ylabel="Density",
                title=f"mean {summary['mean']:+.2f}, std {summary['std']:.2f}, "
                      f"|z|>2: {summary['outside_2']:.0%} (4.6% expected)")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.2)
    # Standard residual plot: z in units of σ against the prediction, ±1/2/3σ bands.
    for k, color, alpha in ((3, "#d62728", 0.15), (2, "#ffd700", 0.35), (1, "#2ca02c", 0.35)):
        inside = np.mean(np.abs(z) <= k)
        ax_rate.axhspan(-k, k, color=color, alpha=alpha, lw=0,
                        label=f"±{k}σ: {inside:.0%} ({[68.3, 95.4, 99.7][k - 1]}%)")
    ax_rate.scatter(predicted, z, s=8, color="black", alpha=0.6, label="voxels", zorder=3)
    ax_rate.axhline(0, color="black", lw=1)
    ax_rate.set(xlabel=rate_label, ylabel="z [σ]", title="z vs predicted rate",
                ylim=(-limit, limit))
    ax_rate.legend(fontsize=8)
    ax_rate.grid(alpha=0.2)
    fig.suptitle(title)
    out_path = Path(out_path)
    for suffix in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{suffix}"), dpi=150)
    plt.close(fig)
    return summary
