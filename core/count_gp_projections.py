"""Saved count-GP adapter for the common spatial projection and plotting engine."""

import hashlib
import json
from pathlib import Path

import numpy as np

from core.projection_adapters import BinomialObservation, projection_adapter
from core.spatial_projections import project_response


def ensure_count_gp_projections(directory, settings):
    """Project a saved count-GP run without retraining or reading simulation files.

    With mixed validation budgets, predictive projections require an explicit N.
    Training overlays include only the same source group as validation overlays.
    """
    from core.binomial_gp import BinomialGP
    from viz.spatial_projections import plot_spatial_projections

    directory = Path(directory)
    with np.load(directory / "train_counts.npz") as data:
        train = {k: data[k] for k in data.files}
    with np.load(directory / "validation_counts.npz") as data:
        valid = {k: data[k] for k in data.files}
    group = "hf" if "hf" in valid["source_group"] else str(valid["source_group"][0])
    selected = valid["source_group"] == group
    if settings.quantity == "observed_fraction" and settings.target_events is None:
        budgets = np.unique(valid["trials"][selected])
        if len(budgets) != 1:
            raise ValueError("Mixed validation budgets: set projections.target_events explicitly")
        settings = settings.model_copy(update={"target_events": int(budgets[0])})
    sources = [directory / n for n in ("model.pkl", "train_counts.npz", "validation_counts.npz")]
    signature = dict(
        version=1,
        backend="binomial_laplace",
        config=settings.model_dump(mode="json"),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    output = directory / (
        "projections_observed_fraction"
        if settings.quantity == "observed_fraction"
        else "projections"
    )
    metadata_path, data_path = output / "metadata.json", output / "projections.npz"
    cached = (
        metadata_path.exists()
        and data_path.exists()
        and json.loads(metadata_path.read_text()).get("signature") == signature
    )
    if not cached:
        model = BinomialGP.load(directory / "model.pkl")
        observation = (
            BinomialObservation(settings.target_events)
            if settings.quantity == "observed_fraction"
            else None
        )
        arrays, metadata = project_response(
            projection_adapter("binomial_laplace", model),
            train["theta"],
            settings,
            observation_model=observation,
        )
        training_selected = train["source_group"] == group
        arrays.update(
            observed_train_theta=train["theta"][training_selected],
            observed_train=(train["hits"] / train["trials"])[training_selected],
            observed_validation_theta=valid["theta"][selected],
            observed_validation=(valid["hits"] / valid["trials"])[selected],
        )
        metadata.update(response_label="Detection probability", observation_group=group.upper())
        output.mkdir(exist_ok=True)
        np.savez_compressed(data_path, **arrays)
        metadata_path.write_text(json.dumps(dict(signature=signature, **metadata), indent=2))
    from core.projection_selection import selected_figures

    names = selected_figures(settings)
    plot_meta = output / "plot_metadata.json"
    stale_labels = (
        not plot_meta.exists() or json.loads(plot_meta.read_text()).get("plot_version") != 3
    )
    missing_counts = (
        settings.quantity == "observed_fraction" and not (output / "coverage_counts.json").exists()
    )
    if (
        not cached
        or stale_labels
        or missing_counts
        or any(not (output / f"{name}.{ext}").exists() for name in names for ext in ("png", "pdf"))
    ):
        plot_spatial_projections(output)
    return output
