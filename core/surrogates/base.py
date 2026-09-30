"""Numpy prediction contract shared by neural and tree components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

from schemas.data_models import StandardBatch


@dataclass(frozen=True)
class EventPrediction:
    """Raw logits [voxel,event]; no inferred Gaussian/epistemic uncertainty."""

    logits: np.ndarray
    legacy_scale: np.ndarray | None = None

    def __post_init__(self):
        value = np.asarray(self.logits, dtype=float)
        if value.ndim != 2 or not value.size or not np.isfinite(value).all():
            raise ValueError("Expected finite nonempty [voxel,event] logits")
        object.__setattr__(self, "logits", value)
        if self.legacy_scale is not None:
            scale = np.asarray(self.legacy_scale, dtype=float)
            if scale.shape != value.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Expected positive finite legacy scale matching logits")
            object.__setattr__(self, "legacy_scale", scale)

    @property
    def probabilities(self):
        return expit(self.logits)

    @property
    def mean(self):
        return self.probabilities.mean(axis=1)


@dataclass(frozen=True)
class Episode:
    context: StandardBatch | list[StandardBatch] | None
    target: StandardBatch | list[StandardBatch]


class EventSurrogate(ABC):
    """Public model interface. Context is required only when uses_context=True."""

    uses_context = False
    supports_uncertainty = False

    def __init__(self, config, dim_theta, dim_phi):
        if not any(d is not None for d in (dim_theta, dim_phi)):
            raise ValueError("At least one input feature group is required")
        if any(d is not None and d < 1 for d in (dim_theta, dim_phi)):
            raise ValueError("Feature dimensions must be positive or None")
        self.config, self.dim_theta, self.dim_phi = config, dim_theta, dim_phi
        self.metadata = {}

    def validate(self, target, context=None):
        for name, dimension in [("theta", self.dim_theta), ("phi", self.dim_phi)]:
            value = getattr(target, name)
            if (value is None) != (dimension is None):
                raise ValueError(f"{name} presence does not match model schema")
            if value is not None and (value.shape[-1] != dimension or not np.isfinite(value).all()):
                raise ValueError(f"{name} has wrong dimension or nonfinite values")
        if self.uses_context:
            if context is None:
                raise ValueError("CNP prediction requires context observations")
            if context.mode != target.mode or context.batch_size != target.batch_size:
                raise ValueError("Context must match target mode and voxel count")
            for name in ("theta", "phi"):
                value, other = getattr(context, name), getattr(target, name)
                if (value is None) != (other is None):
                    raise ValueError("Context feature presence does not match target")
                if value is not None and (
                    value.shape[-1] != other.shape[-1] or not np.isfinite(value).all()
                ):
                    raise ValueError("Context feature schema does not match target")
            if target.theta is not None and not np.array_equal(target.theta, context.theta):
                raise ValueError("Context and target must refer to the same voxel coordinates")

    @abstractmethod
    def predict(self, target: StandardBatch, *, context=None) -> EventPrediction: ...

    def fit(
        self, train, training, *, validation=None, selection="voxel_rate_mae", checkpoints=None
    ):
        from core.surrogates.training import fit_surrogate

        return fit_surrogate(
            self,
            train,
            training,
            validation=validation,
            selection=selection,
            checkpoints=checkpoints,
        )

    def save(self, path: str | Path, *, metadata=None):
        from core.surrogates.checkpoints import save_surrogate

        save_surrogate(path, self, metadata=metadata)
