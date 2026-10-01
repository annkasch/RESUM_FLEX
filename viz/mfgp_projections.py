"""Backward-compatible names for the shared GP projection plots."""

from viz.spatial_projections import (
    plot_projection_counts as plot_projection_counts,
)
from viz.spatial_projections import (
    plot_spatial_projections as plot_mfgp_projections,
)
from viz.spatial_projections import (
    projection_counts as projection_counts,
)

__all__ = ["plot_mfgp_projections", "projection_counts", "plot_projection_counts"]
