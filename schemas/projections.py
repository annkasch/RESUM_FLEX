"""Shared spatial projection settings for any GP backend."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProjectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quantity: Literal["latent_mean", "observed_fraction"] = "latent_mean"
    plots: list[
        Literal["projected_axes", "projected_planes", "marginalized_axes", "marginalized_planes"]
    ] = Field(default_factory=lambda: ["projected_axes", "projected_planes"])
    target_events: int | None = Field(default=None, gt=0)
    enabled: bool = False
    domain: Literal["training_convex_hull", "box"] = "training_convex_hull"
    bounds: list[tuple[float, float]] | None = None
    axis_labels: tuple[str, str, str] = ("x", "y", "z")
    units: str = "m"
    grid_steps: int = Field(default=12, ge=3, le=16)
    n_draws: int = Field(default=8192, ge=2048, le=65536)
    seed: int = 42

    @model_validator(mode="after")
    def valid_bounds(self):
        import math

        if self.domain == "box" and self.bounds is None:
            raise ValueError("Box projections require explicit physical bounds")
        if self.bounds is not None and (
            len(self.bounds) != 3
            or any(
                not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in self.bounds
            )
        ):
            raise ValueError("Projection bounds must contain three finite increasing pairs")
        return self
