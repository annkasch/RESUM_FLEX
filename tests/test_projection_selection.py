from pathlib import Path

import pytest

from core.projection_selection import ensure_selected_projections, selected_figures
from schemas.projections import ProjectionConfig


def test_defaults_generate_only_requested_predictive_figures(monkeypatch):
    calls = []

    def ensure(directory, settings):
        calls.append((settings.quantity, selected_figures(settings)))
        return Path(directory) / settings.quantity

    monkeypatch.setattr("core.count_gp_projections.ensure_count_gp_projections", ensure)
    settings = ProjectionConfig(enabled=True)
    paths = ensure_selected_projections("/tmp/run", settings, backend="binomial_laplace")
    assert calls == [("observed_fraction", ["curves_observed", "planes_observed"])]
    assert [p.name for p in paths] == ["curves_observed.png", "planes_observed.png"]


def test_explicit_mixed_selection_and_disabled(monkeypatch):
    calls = []

    def ensure(directory, settings):
        calls.append((settings.quantity, selected_figures(settings)))
        return Path(directory) / settings.quantity

    monkeypatch.setattr("core.mfgp_projections.ensure_mfgp_projections", ensure)
    settings = ProjectionConfig(enabled=True, plots=["marginalized_axes", "projected_planes"])
    paths = ensure_selected_projections("/tmp/run", settings, backend="mfgp")
    assert calls == [("latent_mean", ["curves"]), ("observed_fraction", ["planes_observed"])]
    assert [p.name for p in paths] == ["curves.png", "planes_observed.png"]
    assert ensure_selected_projections("/tmp/run", ProjectionConfig(), backend="mfgp") == []
    assert (
        ensure_selected_projections(
            "/tmp/run", ProjectionConfig(enabled=True, plots=[]), backend="mfgp"
        )
        == []
    )
    with pytest.raises(ValueError):
        ProjectionConfig(plots=["typo"])


def test_renderer_only_saves_selected_figures(tmp_path, monkeypatch):
    import json
    from itertools import product

    import numpy as np
    from matplotlib.figure import Figure

    from core.projection_adapters import BinomialObservation, GaussianResponse
    from core.spatial_projections import project_response
    from viz.spatial_projections import plot_spatial_projections

    class Adapter:
        def prepare(self, x):
            return GaussianResponse(np.full(len(x), 0.01), np.eye(len(x)) * 1e-12, "identity")

    settings = ProjectionConfig(
        quantity="observed_fraction", grid_steps=3, n_draws=2048, target_events=5000
    )
    x = np.array(list(product((0.0, 1.0), repeat=3)))
    arrays, meta = project_response(
        Adapter(), x, settings, observation_model=BinomialObservation(5000)
    )
    arrays.update(
        observed_train_theta=x,
        observed_train=np.full(8, 0.01),
        observed_validation_theta=x,
        observed_validation=np.full(8, 0.01),
    )
    np.savez(tmp_path / "projections.npz", **arrays)
    (tmp_path / "metadata.json").write_text(
        json.dumps(dict(signature={"config": settings.model_dump()}, **meta))
    )
    saved = []
    monkeypatch.setattr(
        Figure, "savefig", lambda self, path, **kwargs: saved.append(Path(path).name)
    )
    plot_spatial_projections(tmp_path)
    assert set(saved) == {
        "curves_observed.png",
        "curves_observed.pdf",
        "planes_observed.png",
        "planes_observed.pdf",
    }
