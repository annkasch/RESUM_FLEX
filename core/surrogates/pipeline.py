"""Surrogate-to-MFGP array boundary without importing GP dependencies."""

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
        if batch.theta is None:
            raise ValueError("MFGP requires design coordinates theta")
        nc = int(batch.events_per_voxel.min()) // 2 if nc is None else nc
        context, target = split_context_target(batch, nc, seed=seed + offset)
        result[f"X_{fidelity}"] = batch.theta
        result[f"Y_{fidelity}_cnp"] = model.predict(target, context=context).mean[:, None]
        if fidelity == "hf":
            result["Y_hf_raw"] = target.voxel_mean(target.labels)[:, None]
            # Per-voxel target counts: raw fractions have voxel-dependent binomial noise.
            result["hf_target_events"] = target.events_per_voxel[:, None]
    return result
