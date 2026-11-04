"""Laplace posterior of a final linear Bernoulli layer (numpy only).

The network's last layer computes ``logit = w · f̃`` with ``f̃ = [f, 1]`` (features of the
layer before plus a bias input). With the trained weights ``w*`` as the posterior mode and
an isotropic Gaussian prior N(0, λ⁻¹ I), the Laplace approximation is

    w ~ N(w*, (H + λ I)⁻¹),      H = Σ_i p_i (1 - p_i) f̃_i f̃_iᵀ,

which is the exact Hessian of the Bernoulli negative log-likelihood in the last layer
(summed over all natural training events, p_i = sigmoid(w* · f̃_i)). λ is either fixed or
chosen by maximizing the Laplace log marginal likelihood (evidence) of the training data,

    log Z(λ) = const + (d/2) log λ - (λ/2) |w*|² - ½ log det(H + λ I).

The network was trained by maximum likelihood, so the prior is post hoc; ``w*`` is kept as
the mode for every λ. Uncertainty from the frozen feature layers is not included.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

# log10 λ search range for the evidence maximization.
LOG10_PRIOR_RANGE = (-8.0, 8.0)


@dataclass(frozen=True)
class LastLayerPosterior:
    """Gaussian posterior over the last-layer weights [weights..., bias].

    mean: [d] trained weights w*; eigenvalues, eigenvectors: decomposition of H (d = H + 1
    features); prior_precision: λ; events: training events in H.
    """

    mean: np.ndarray  # [d]
    eigenvalues: np.ndarray  # [d] of H, >= 0
    eigenvectors: np.ndarray  # [d, d] columns
    prior_precision: float
    log_evidence: float
    events: int

    def __post_init__(self):
        d = len(self.mean)
        if self.mean.ndim != 1 or self.eigenvectors.shape != (d, d) or len(self.eigenvalues) != d:
            raise ValueError("Posterior mean, eigenvalues and eigenvectors must agree in size")
        if not self.prior_precision > 0 or not np.isfinite(self.prior_precision):
            raise ValueError("prior_precision must be positive and finite")
        if np.any(self.eigenvalues < 0) or not np.isfinite(self.eigenvalues).all():
            raise ValueError("Hessian eigenvalues must be finite and nonnegative")

    @property
    def covariance(self) -> np.ndarray:  # [d, d]
        scale = 1.0 / (self.eigenvalues + self.prior_precision)
        return (self.eigenvectors * scale) @ self.eigenvectors.T

    def logit_variance(self, features: np.ndarray) -> np.ndarray:
        """Var(w · f̃) for features [..., d] (bias column included)."""
        projected = features @ self.eigenvectors  # [..., d]
        return (projected**2 / (self.eigenvalues + self.prior_precision)).sum(axis=-1)

    def sample(self, n_draws: int, seed: int) -> np.ndarray:
        """Weight draws [d, n_draws] from N(w*, (H + λ I)⁻¹)."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        z = rng.standard_normal((len(self.mean), n_draws))
        scale = 1.0 / np.sqrt(self.eigenvalues + self.prior_precision)
        return self.mean[:, None] + self.eigenvectors @ (scale[:, None] * z)

    def to_arrays(self) -> dict:
        return dict(
            mean=self.mean,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            prior_precision=np.array(self.prior_precision),
            log_evidence=np.array(self.log_evidence),
            events=np.array(self.events),
        )

    @classmethod
    def from_arrays(cls, arrays) -> "LastLayerPosterior":
        return cls(
            mean=np.asarray(arrays["mean"], float),
            eigenvalues=np.asarray(arrays["eigenvalues"], float),
            eigenvectors=np.asarray(arrays["eigenvectors"], float),
            prior_precision=float(arrays["prior_precision"]),
            log_evidence=float(arrays["log_evidence"]),
            events=int(arrays["events"]),
        )


def log_evidence(prior_precision: float, eigenvalues: np.ndarray, mean: np.ndarray) -> float:
    """Laplace log marginal likelihood up to the λ-independent likelihood term."""
    lam = float(prior_precision)
    d = len(mean)
    return float(
        0.5 * d * np.log(lam) - 0.5 * lam * mean @ mean - 0.5 * np.log(eigenvalues + lam).sum()
    )


def laplace_posterior(
    hessian: np.ndarray,  # [d, d] Σ p(1-p) f̃ f̃ᵀ
    mean: np.ndarray,  # [d] trained weights w*
    *,
    prior_precision: float | str = "evidence",
    events: int,
) -> LastLayerPosterior:
    """Posterior from the accumulated Hessian; λ fixed or by evidence maximization."""
    hessian, mean = np.asarray(hessian, float), np.asarray(mean, float)
    if hessian.shape != (len(mean), len(mean)) or not np.isfinite(hessian).all():
        raise ValueError("Hessian must be a finite square matrix matching the weights")
    eigenvalues, eigenvectors = np.linalg.eigh((hessian + hessian.T) / 2)
    tolerance = 1e-10 * max(1.0, float(np.abs(eigenvalues).max()))
    if eigenvalues.min() < -tolerance:
        raise ValueError("Hessian is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if prior_precision == "evidence":
        if not np.any(mean):
            raise ValueError("Evidence maximization needs nonzero trained weights")

        def objective(log10_lam):
            return -log_evidence(10.0**log10_lam, eigenvalues, mean)

        # Coarse grid then bounded refinement: the objective is smooth but may be flat.
        grid = np.linspace(*LOG10_PRIOR_RANGE, 65)
        best = grid[np.argmin([objective(x) for x in grid])]
        step = grid[1] - grid[0]
        low, high = max(best - step, grid[0]), min(best + step, grid[-1])
        result = minimize_scalar(objective, bounds=(low, high), method="bounded")
        lam = float(10.0**result.x)
    else:
        lam = float(prior_precision)
    return LastLayerPosterior(
        mean=mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        prior_precision=lam,
        log_evidence=log_evidence(lam, eigenvalues, mean),
        events=int(events),
    )
