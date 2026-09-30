"""Surrogate-to-MFGP array boundary without importing GP dependencies."""

import numpy as np
from data.grouped_batches import batch_groups
from core.surrogate_cnp import split_context_target


def prepare_surrogate_datasets(
    model, lf_batch, hf_batch, *, n_lf_context=None, n_hf_context=None, seed=0
):
    """Return the existing fit_mfgp_three_fidelity input keys for any surrogate.

    Historical *_cnp key names are retained for GP compatibility. All quantities
    average over the same target events; no classifier uncertainty is invented.
    """
    result = {}
    for fidelity, batch, nc, offset in [
        ("lf", lf_batch, n_lf_context, 1),
        ("hf", hf_batch, n_hf_context, 101),
    ]:
        positions, means, raw = [], [], []
        for group in batch_groups(batch):
            if group.theta is None:
                raise ValueError("MFGP requires design coordinates theta")
            count = group.n_events // 2 if nc is None else nc
            context, target = split_context_target(group, count, seed=seed + offset)
            positions.append(group.theta)
            means.append(model.predict(target, context=context).mean[:, None])
            raw.append(target.labels.mean(axis=1)[:, None])
        result[f"X_{fidelity}"] = np.concatenate(positions)
        result[f"Y_{fidelity}_cnp"] = np.concatenate(means)
        if fidelity == "hf":
            result["Y_hf_raw"] = np.concatenate(raw)
    return result


def mfgp_level_arrays(data, lf_event_counts, *, lf_levels="pooled"):
    """Arrange the same observations into pooled or event-count LF levels."""
    counts = np.asarray(lf_event_counts)
    if counts.shape != (len(data["X_lf"]),):
        raise ValueError("LF event counts must identify every LF training row")
    if lf_levels == "pooled":
        xs, ys = [data["X_lf"]], [data["Y_lf_cnp"]]
        names = ["LF surrogate mean"]
        mapping = {int(n): 0 for n in np.unique(counts)}
    elif lf_levels == "by_event_count":
        sizes = sorted(np.unique(counts))
        xs = [data["X_lf"][counts == n] for n in sizes]
        ys = [data["Y_lf_cnp"][counts == n] for n in sizes]
        names = [f"LF{int(n)} surrogate mean" for n in sizes]
        mapping = {int(n): i for i, n in enumerate(sizes)}
    else:
        raise ValueError(f"Unknown LF level strategy: {lf_levels}")
    xs.extend([data["X_hf"], data["X_hf"]])
    ys.extend([data["Y_hf_cnp"], data["Y_hf_raw"]])
    names.extend(["HF surrogate mean", "HF raw target fraction"])
    return xs, ys, names, mapping
