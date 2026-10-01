"""Config-driven projection selection shared by saved runs and notebooks."""

PLOTS = {
    "projected_axes": ("observed_fraction", "curves_observed"),
    "projected_planes": ("observed_fraction", "planes_observed"),
    "marginalized_axes": ("latent_mean", "curves"),
    "marginalized_planes": ("latent_mean", "planes"),
}


def selected_figures(settings):
    return list(
        dict.fromkeys(
            PLOTS[name][1] for name in settings.plots if PLOTS[name][0] == settings.quantity
        )
    )


def ensure_selected_projections(directory, settings, *, backend):
    """Generate selected quantities only and return ordered PNG paths for display."""
    if not settings.enabled or not settings.plots:
        return []
    if backend == "binomial_laplace":
        from core.count_gp_projections import ensure_count_gp_projections as ensure
    elif backend == "mfgp":
        from core.mfgp_projections import ensure_mfgp_projections as ensure
    else:
        raise ValueError(f"Unsupported saved-run projection backend: {backend}")
    outputs = {}
    for name in settings.plots:
        quantity, _ = PLOTS[name]
        if quantity not in outputs:
            outputs[quantity] = ensure(
                directory, settings.model_copy(update={"quantity": quantity})
            )
    return [
        outputs[PLOTS[name][0]] / f"{PLOTS[name][1]}.png" for name in dict.fromkeys(settings.plots)
    ]
