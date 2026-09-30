"""Single-population spatial GP with integer binomial observations.

Optional GPy backend, Laplace posterior approximation, logistic probability link.
No fidelity input, pseudo-count targets, CNP or Gaussian observation noise.
"""

import pickle
from pathlib import Path

import GPy
import numpy as np
from GPy.util.linalg import jitchol
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import expit, gammaln


class LogisticLink(GPy.likelihoods.link_functions.GPTransformation):
    def transf(self, f):
        return expit(f)

    def dtransf_df(self, f):
        p = expit(f)
        return p * (1 - p)

    def d2transf_df2(self, f):
        p = expit(f)
        return p * (1 - p) * (1 - 2 * p)

    def d3transf_df3(self, f):
        p = expit(f)
        return p * (1 - p) * (1 - 6 * p + 6 * p * p)


class BinomialLogit(GPy.likelihoods.Binomial):
    """Stable binomial log likelihood and analytic latent derivatives.

    GPy's generic link chain rule divides by p(1-p); direct logit derivatives
    avoid numerical singularities for very rare events or extreme logits.
    """

    def __init__(self):
        super().__init__(gp_link=LogisticLink())
        self.log_concave = True

    def logpdf(self, f, y, Y_metadata=None):
        n = Y_metadata["trials"]
        constant = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
        return constant - y * np.logaddexp(0, -f) - (n - y) * np.logaddexp(0, f)

    def dlogpdf_df(self, f, y, Y_metadata=None):
        return y - Y_metadata["trials"] * expit(f)

    def d2logpdf_df2(self, f, y, Y_metadata=None):
        p = expit(f)
        return -Y_metadata["trials"] * p * (1 - p)

    def d3logpdf_df3(self, f, y, Y_metadata=None):
        p = expit(f)
        return -Y_metadata["trials"] * p * (1 - p) * (1 - 2 * p)


class CountLaplace(GPy.inference.latent_function_inference.Laplace):
    """Damped Newton mode search for the concave binomial-logit posterior.

    Backtracking avoids GPy's unbounded Brent search failing on flat objectives
    near convergence. The posterior and evidence calculations remain GPy's.
    """

    def rasm_mode(self, K, Y, likelihood, Ki_f_init, Y_metadata=None, *args, **kwargs):
        L = jitchol(K)
        u = np.zeros_like(Y)

        def objective(u):
            return float(
                -0.5 * np.sum(u * u) + likelihood.logpdf(L @ u, Y, Y_metadata=Y_metadata).sum()
            )

        for _ in range(self._mode_finding_max_iter):
            f = L @ u
            grad = L.T @ likelihood.dlogpdf_df(f, Y, Y_metadata=Y_metadata) - u
            W = -likelihood.d2logpdf_df2(f, Y, Y_metadata=Y_metadata)
            hessian = np.eye(len(Y)) + L.T @ (W * L)
            direction = cho_solve(cho_factor(hessian), grad)
            if float(np.sum(grad * direction)) < 1e-9:
                self.bad_fhat = False
                return f, solve_triangular(L.T, u, lower=False)
            old = objective(u)
            step = 1.0
            for _ in range(60):
                candidate = u + step * direction
                new = objective(candidate)
                if np.isfinite(new) and new >= old - 1e-10:
                    break
                step *= 0.5
            else:
                raise RuntimeError("Binomial posterior mode line search failed")
            u = candidate
        raise RuntimeError("Binomial posterior mode did not converge")


def counts_array(values, size, *, positive=False):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(size, values)
    values = values.reshape(-1)
    if len(values) != size or not np.isfinite(values).all():
        raise ValueError("Counts must be finite and match the number of coordinates")
    if np.any(values != np.floor(values)) or np.any(values < (1 if positive else 0)):
        raise ValueError("Counts must be integers; trials must be positive and hits nonnegative")
    return values.astype(np.int64)


class BinomialGP:
    backend = "binomial_laplace"

    def __init__(self, *, kernel="matern52"):
        if kernel not in ("rbf", "matern52"):
            raise ValueError("Unknown kernel")
        self.kernel_name = kernel
        self.model = None

    def fit(self, theta, hits, trials, *, max_iters=300, n_restarts=2, seed=42):
        x = np.asarray(theta, dtype=float)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Coordinates must be a nonempty finite matrix")
        m = counts_array(hits, len(x))
        n = counts_array(trials, len(x), positive=True)
        if np.any(m > n):
            raise ValueError("Hit counts cannot exceed trials")
        # Coordinates are the only covariates. N enters the likelihood only.
        self.offset = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale == 0] = 1
        normalized = (x - self.offset) / self.scale
        factory = GPy.kern.RBF if self.kernel_name == "rbf" else GPy.kern.Matern52
        kernel = factory(x.shape[1], ARD=True) + GPy.kern.Bias(x.shape[1], variance=25.0)
        # Broad finite optimization bounds on standardized-coordinate hyperparameters.
        kernel.constrain_bounded(1e-4, 1e4, warning=False)
        inference = CountLaplace()
        inference._mode_finding_max_iter = 100
        inference._mode_finding_tolerance = 1e-7
        self.model = GPy.core.GP(
            normalized,
            m[:, None].astype(float),
            kernel=kernel,
            likelihood=BinomialLogit(),
            inference_method=inference,
            Y_metadata={"trials": n[:, None].astype(float)},
        )
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            self.model.optimize_restarts(
                num_restarts=n_restarts, max_iters=max_iters, verbose=False
            )
        finally:
            np.random.set_state(state)
        if not np.isfinite(self.model.log_likelihood()):
            raise ValueError("Nonfinite fitted Laplace evidence")
        self.training_trials = n
        self.training_hits = m
        return self

    def predict_latent(self, theta, *, full_cov=False):
        if self.model is None:
            raise RuntimeError("Fit the GP first")
        x = np.asarray(theta, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.offset) or not np.isfinite(x).all():
            raise ValueError("Query coordinates must match training dimensions and be finite")
        mean, variance = self.model.predict(
            (x - self.offset) / self.scale, full_cov=full_cov, include_likelihood=False
        )
        if full_cov:
            variance = np.asarray(variance).reshape(len(x), len(x))
            variance = (variance + variance.T) / 2
        else:
            variance = np.maximum(variance.ravel(), 0)
        return mean.ravel(), variance

    def predict(self, theta):
        """Probability mean and epistemic variance using Gaussian quadrature."""
        mu, variance = self.predict_latent(theta)
        nodes, weights = hermgauss(64)
        p = expit(mu[:, None] + np.sqrt(2 * variance[:, None]) * nodes)
        mean = p @ weights / np.sqrt(np.pi)
        second = (p * p) @ weights / np.sqrt(np.pi)
        return mean, np.maximum(second - mean * mean, 0)

    def predict_observations(self, theta, trials, *, n_draws=32768, seed=42):
        """Joint latent draws followed by binomial counts at exact coordinates."""
        mu, covariance = self.predict_latent(theta, full_cov=True)
        n = counts_array(trials, len(mu), positive=True)
        rng = np.random.default_rng(seed)
        # PSD eigenfactor handles duplicate coordinates without adding count noise.
        eigenvalues, vectors = np.linalg.eigh(covariance)
        if eigenvalues.min() < -1e-7 * max(1.0, eigenvalues.max()):
            raise ValueError("Invalid posterior covariance")
        factor = vectors * np.sqrt(np.maximum(eigenvalues, 0))
        p = expit(mu + rng.standard_normal((n_draws, len(mu))) @ factor.T)
        return rng.binomial(n, p) / n

    def save(self, path):
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted GP")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream)
        return path

    @classmethod
    def load(cls, path):
        """Load only trusted local checkpoints."""
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls) or model.model is None:
            raise ValueError("Not a fitted binomial GP checkpoint")
        return model
