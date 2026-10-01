"""Backend adapters for joint physical-response draws, excluding observation noise."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import cholesky
from scipy.special import expit


class JointResponse(Protocol):
    mean: np.ndarray
    metadata: dict

    def sample(self, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """Return (draws, locations), preserving spatial correlations."""
        ...


class ProjectionAdapter(Protocol):
    def prepare(self, coordinates: np.ndarray) -> JointResponse:
        """Prepare joint response at physical coordinates, without observation noise."""
        ...


class ObservationModel(Protocol):
    metadata: dict

    def sample(self, responses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw observed responses of the same shape from latent responses."""
        ...


@dataclass
class GaussianResponse:
    latent_mean: np.ndarray
    covariance: np.ndarray
    link: str

    def __post_init__(self):
        mean = np.asarray(self.latent_mean, float).reshape(-1)
        covariance = np.asarray(self.covariance, float)
        if (
            covariance.shape != (len(mean), len(mean))
            or not np.isfinite(mean).all()
            or not np.isfinite(covariance).all()
        ):
            raise ValueError("Nonfinite or incorrectly shaped joint posterior")
        covariance = (covariance + covariance.T) / 2
        diagonal = np.diag(covariance)
        jitter = 0.0
        for attempt in range(8):
            try:
                self.factor = cholesky(covariance + jitter * np.eye(len(mean)), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter = max(float(np.max(np.abs(diagonal))), 1.0) * 10.0 ** (-12 + attempt)
        else:
            raise ValueError("Joint posterior covariance is not positive semidefinite")
        self.latent_mean = mean
        variance = np.maximum(diagonal, 0)
        with np.errstate(over="raise", invalid="raise"):
            if self.link == "log":
                self.mean = np.exp(mean + variance / 2)
            elif self.link == "logit":
                nodes, weights = hermgauss(64)
                self.mean = (
                    expit(mean[:, None] + np.sqrt(2 * variance[:, None]) * nodes)
                    @ weights
                    / np.sqrt(np.pi)
                )
            elif self.link == "identity":
                self.mean = mean.copy()
            else:
                raise ValueError(f"Unknown response link: {self.link}")
        self.metadata = dict(
            cholesky_jitter=jitter,
            transform=self.link,
            posterior="joint latent; observation noise excluded",
        )

    def sample(self, n_draws, rng):
        draws = self.latent_mean + rng.standard_normal((n_draws, len(self.mean))) @ self.factor.T
        with np.errstate(over="raise", invalid="raise"):
            if self.link == "log":
                return np.exp(draws)
            if self.link == "logit":
                return expit(draws)
        return draws


class MFGPProjectionAdapter:
    def __init__(self, model, *, offset=None, scale=None, fidelity=None):
        self.model, self.fidelity = model, fidelity
        self.offset = np.zeros(3) if offset is None else np.asarray(offset, float)
        self.scale = np.ones(3) if scale is None else np.asarray(scale, float)
        if (
            self.offset.shape != (3,)
            or self.scale.shape != (3,)
            or not np.isfinite([self.offset, self.scale]).all()
            or np.any(self.scale <= 0)
        ):
            raise ValueError("Projections require a valid three-coordinate affine normalization")

    def prepare(self, coordinates):
        query = (coordinates - self.offset) / self.scale
        kwargs = {} if self.fidelity is None else {"fidelity": self.fidelity}
        mean, covariance = self.model.predict_joint_transformed(query, **kwargs)
        response = GaussianResponse(
            mean, covariance, getattr(self.model, "output_transform", "identity")
        )
        response.metadata.update(backend="mfgp", fidelity=self.fidelity)
        return response


class BinomialGPProjectionAdapter:
    def __init__(self, model):
        self.model = model

    def prepare(self, coordinates):
        # This backend owns its physical-coordinate normalization.
        mean, covariance = self.model.predict_latent(coordinates, full_cov=True)
        response = GaussianResponse(mean, covariance, "logit")
        response.metadata.update(backend="binomial_laplace")
        return response


_ADAPTERS = {"mfgp": MFGPProjectionAdapter, "binomial_laplace": BinomialGPProjectionAdapter}


def register_projection_adapter(name, factory):
    """Register a factory(model, **options) for a new backend; no engine edits needed."""
    if name in _ADAPTERS:
        raise ValueError(f"Projection adapter already registered: {name}")
    _ADAPTERS[name] = factory


def projection_adapter(name, model, **options):
    try:
        factory = _ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"No projection adapter registered for {name}") from exc
    return factory(model, **options)


@dataclass
class BinomialObservation:
    trials: int

    def __post_init__(self):
        if not isinstance(self.trials, (int, np.integer)) or self.trials <= 0:
            raise ValueError("Binomial projections require positive integer trials")

    def sample(self, probabilities, rng):
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError(
                "GP latent draws fall outside [0,1]; cannot use as binomial probabilities. "
                "Predictions are not silently clipped."
            )
        return rng.binomial(self.trials, probabilities) / self.trials

    @property
    def metadata(self):
        return dict(
            observation_model="binomial",
            target_events=int(self.trials),
            noise_description="binomial counting noise",
        )
