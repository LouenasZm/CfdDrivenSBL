r"""Semi-analytical model evidence of nonlinear sparse Bayesian learning.

Given the Gaussian-mixture approximation

.. math::

    p(\mathcal{D}\mid\phi)\,p(\phi_{-\alpha})
    \approx \sum_k a_k\,\mathcal{N}(\phi\mid\mu_k,\Sigma_k)

and Gaussian ARD priors :math:`\mathcal{N}(\phi_\alpha \mid 0, A^{-1})`,
:math:`A = \mathrm{diag}(\alpha)`, on the questionable parameters, this module
evaluates, as functions of :math:`\theta = \log\alpha`,

* the objective :math:`\mathcal{L}(\theta) = \log\hat p(\mathcal{D}\mid\theta) + \log p(\theta)`
  (Eq. 15 of Sandhu et al., 2021),
* its gradient (Eq. 19) and Hessian (Eq. 21),
* the relevance indicators (Eqs. 22-23) and the posterior mixture (Eq. 12).

Numerical formulation
---------------------
The paper writes the gradient and Hessian with the posterior moments
:math:`P_k = (\Sigma_k^{-1} + A)^{-1}` and :math:`m_k = P_k \Sigma_k^{-1}\mu_k`.
Evaluated literally these lose all their digits once a parameter is pruned
(:math:`\alpha_i \to \infty`), which is exactly the regime the optimiser lives in.
Everything here is expressed through :math:`B_k^{-1} = (\Sigma_k + A^{-1})^{-1}`
instead, using the identities

.. math::

    \gamma_{k,i} = 1 - \alpha_i P_{k,ii} = (B_k^{-1})_{ii} / \alpha_i, \qquad
    \alpha \odot m_k = B_k^{-1}\mu_k, \qquad
    A P_k A = A - B_k^{-1},

none of which involves a cancellation, and none of which requires
:math:`\Sigma_k^{-1}`.  The cost is one Cholesky factorisation per *distinct*
covariance, so a kernel density estimate with thousands of kernels costs a single
factorisation per evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import erf, logsumexp

from .mixture import LOG_2PI, GaussianMixture

__all__ = ["ARDEvidence", "Evaluation", "GammaHyperprior", "truncation_normaliser"]

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


class GammaHyperprior:
    r"""Independent Gamma hyperpriors :math:`\alpha_i \sim \mathcal{G}(r_i, s_i)` (Eq. 6).

    In terms of :math:`\theta_i = \log\alpha_i` the log-density is
    :math:`r_i\theta_i - s_i e^{\theta_i}` up to a constant (Eq. 14).  The default,
    :math:`r = s = 0`, is Jeffreys' prior: flat in :math:`\log\alpha`, so that the
    optimum is dictated by the evidence alone (type-II maximum likelihood).

    Parameters
    ----------
    shape, rate : float or array_like
        :math:`r` and :math:`s`.  A positive rate caps the precision a pruned
        parameter can reach, at a value that depends on the units of the
        parameter; see ``standardize`` in :class:`sblcfd.NSBL`.
    """

    def __init__(self, shape: float | np.ndarray = 0.0, rate: float | np.ndarray = 0.0) -> None:
        self.shape = np.asarray(shape, dtype=float)
        self.rate = np.asarray(rate, dtype=float)
        if np.any(self.shape < 0.0) or np.any(self.rate < 0.0):
            raise ValueError("the shape and rate of the hyperprior must be non-negative")

    @classmethod
    def none(cls) -> GammaHyperprior:
        """No hyperprior at all (the default): pure type-II maximum likelihood."""
        return cls(0.0, 0.0)

    @classmethod
    def paper(cls) -> GammaHyperprior:
        r"""The near-Jeffreys choice :math:`\log r = \log s = -10` of Tables 3-4 of the paper."""
        return cls(np.exp(-10.0), np.exp(-10.0))

    def terms(self, theta: np.ndarray, alpha: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Log-density and its first and second derivatives with respect to ``theta``."""
        value = float(np.sum(self.shape * theta - self.rate * alpha))
        grad = self.shape - self.rate * alpha
        hess = -self.rate * alpha
        return value, np.broadcast_to(grad, theta.shape), np.broadcast_to(hess, theta.shape)


def truncation_normaliser(
    theta: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Mass of :math:`\mathcal{N}(0, e^{-\theta})` on ``[lower, upper]`` and its derivatives.

    Returns :math:`\log Z`, :math:`\partial\log Z/\partial\theta` and
    :math:`\partial^2\log Z/\partial\theta^2`, with
    :math:`Z = \Phi(u\sqrt\alpha) - \Phi(l\sqrt\alpha)`, element-wise.  The interval
    must contain zero.
    """
    t = np.exp(0.5 * theta)
    fin_u, fin_l = np.isfinite(upper), np.isfinite(lower)
    u = np.where(fin_u, upper, 0.0)
    l = np.where(fin_l, lower, 0.0)
    # erf is odd, so with l <= 0 <= u this is a sum of two non-negative terms
    z = 0.5 * (
        np.where(fin_u, erf(u * t / _SQRT2), 1.0) + np.where(fin_l, erf(-l * t / _SQRT2), 1.0)
    )
    pdf_u = np.where(fin_u, _INV_SQRT_2PI * np.exp(-0.5 * (u * t) ** 2), 0.0)
    pdf_l = np.where(fin_l, _INV_SQRT_2PI * np.exp(-0.5 * (l * t) ** 2), 0.0)
    num = 0.5 * t * (u * pdf_u - l * pdf_l)
    dnum = 0.25 * t * (u * pdf_u * (1.0 - (u * t) ** 2) - l * pdf_l * (1.0 - (l * t) ** 2))
    g1 = num / z
    return np.log(z), g1, dnum / z - g1 * g1


class Evaluation(NamedTuple):
    """Objective and derivatives at one value of ``theta = log(alpha)``."""

    value: float
    gradient: np.ndarray | None
    hessian: np.ndarray | None
    log_evidence: float
    weights: np.ndarray


class ARDEvidence:
    r"""NSBL objective :math:`\mathcal{L}(\log\alpha)` for a Gaussian-mixture approximation.

    Parameters
    ----------
    mixture : GaussianMixture
        Approximation of ``likelihood x known prior`` over *all* parameters.
    ard_index : sequence of int, optional
        Coordinates carrying an ARD prior (the questionable parameters
        :math:`\phi_\alpha`).  Defaults to all of them, the classical SBL setting.
    hyperprior : GammaHyperprior, optional
        Prior on the precisions; flat in :math:`\log\alpha` by default.
    lower, upper : array_like, shape (n_ard,), optional
        Bounds of the questionable parameters, used only when ``truncate`` is set.
    truncate : bool
        If ``True`` the ARD priors are zero-mean Gaussians *truncated and
        renormalised* on ``[lower, upper]``.  The evidence then tends to that of a
        uniform prior on the box when :math:`\alpha \to 0`, instead of vanishing.
        If ``False`` (the formulation of the paper) the priors are plain Gaussians.
    theta_max : float
        Half-width of the admissible interval of ``theta``.  A quadratic barrier
        confines ``theta`` to ``theta_shift +- theta_max``, which keeps every
        intermediate quantity representable; its upper end is the pruned state.
    theta_shift : array_like, shape (n_ard,), optional
        Centre of the admissible interval.  Setting it to minus the logarithm of
        the variance of each parameter makes the interval, and with it the pruned
        state, independent of the units of the parameters.  Defaults to zero.
    """

    def __init__(
        self,
        mixture: GaussianMixture,
        ard_index: Sequence[int] | None = None,
        *,
        hyperprior: GammaHyperprior | None = None,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        truncate: bool = False,
        theta_max: float = 30.0,
        theta_shift: np.ndarray | None = None,
    ) -> None:
        self.mixture = mixture
        if ard_index is None:
            ard_index = np.arange(mixture.dim)
        self.ard_index = np.asarray(ard_index, dtype=int)
        if self.ard_index.size == 0:
            raise ValueError("at least one parameter must carry an ARD prior")
        self.n = self.ard_index.size
        self.hyperprior = GammaHyperprior() if hyperprior is None else hyperprior
        self.theta_max = float(theta_max)
        self.theta_shift = (
            np.zeros(self.n)
            if theta_shift is None
            else np.broadcast_to(np.asarray(theta_shift, dtype=float), (self.n,)).copy()
        )
        self.theta_lower = self.theta_shift - self.theta_max
        self.theta_upper = self.theta_shift + self.theta_max
        self.truncate = bool(truncate)

        self.lower = np.full(self.n, -np.inf) if lower is None else np.asarray(lower, dtype=float)
        self.upper = np.full(self.n, np.inf) if upper is None else np.asarray(upper, dtype=float)
        if self.truncate and (np.any(self.lower > 0.0) or np.any(self.upper < 0.0)):
            raise ValueError("a truncated zero-mean ARD prior needs lower <= 0 <= upper")

        marginal = mixture.marginal(self.ard_index)
        self._mu = marginal.means
        self._cov = marginal.covariances
        self._logw = marginal.log_weights
        self._groups = list(marginal.groups())

    # ------------------------------------------------------------------ #
    #  core evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, theta: np.ndarray, order: int = 2) -> Evaluation:
        """Objective (to be maximised) with its gradient (``order >= 1``) and Hessian (``2``)."""
        theta = np.asarray(theta, dtype=float)
        th = np.clip(theta, self.theta_lower, self.theta_upper)
        alpha, inv_alpha = np.exp(th), np.exp(-th)
        n, n_kernels = self.n, self._mu.shape[0]
        eye = np.eye(n)

        log_kernel = np.empty(n_kernels)
        solved = np.empty((n_kernels, n)) if order >= 1 else None  # rows are B^-1 mu_k
        b_inverse: dict[int, np.ndarray] = {}
        for g, sl in self._groups:
            chol = cholesky(self._cov[g] + np.diag(inv_alpha), lower=True, check_finite=False)
            logdet = 2.0 * np.log(np.diag(chol)).sum()
            mu = self._mu[sl]
            if order >= 1:
                binv = cho_solve((chol, True), eye, check_finite=False)
                binv = 0.5 * (binv + binv.T)
                solved[sl] = mu @ binv
                quad = np.einsum("ki,ki->k", mu, solved[sl])
                b_inverse[g] = binv
            else:
                z = solve_triangular(chol, mu.T, lower=True, check_finite=False)
                quad = np.einsum("ij,ij->j", z, z)
            log_kernel[sl] = -0.5 * (n * LOG_2PI + logdet + quad)

        logw = self._logw + log_kernel
        log_evidence = float(logsumexp(logw))
        weights = np.exp(logw - log_evidence)

        hp_value, hp_grad, hp_hess = self.hyperprior.terms(th, alpha)
        value = log_evidence + hp_value
        if self.truncate:
            log_z, tr_grad, tr_hess = truncation_normaliser(th, self.lower, self.upper)
            log_evidence -= float(log_z.sum())
            value -= float(log_z.sum())

        excess = theta - th
        outside = excess != 0.0
        value -= 0.5 * float(excess @ excess)
        if order == 0:
            return Evaluation(value, None, None, log_evidence, weights)

        # v_{k,i} = d log N(mu_k | 0, B_k) / d theta_i = (gamma_{k,i} - alpha_i m_{k,i}^2) / 2
        v = np.empty((n_kernels, n))
        for g, sl in self._groups:
            gamma = np.diag(b_inverse[g]) * inv_alpha
            v[sl] = 0.5 * (gamma[None, :] - solved[sl] ** 2 * inv_alpha[None, :])
        v_bar = weights @ v
        grad = v_bar + hp_grad
        if self.truncate:
            grad = grad - tr_grad
        grad = np.where(outside, -excess, grad)
        if order == 1:
            return Evaluation(value, grad, None, log_evidence, weights)

        hess = (v * weights[:, None]).T @ v - np.outer(v_bar, v_bar)
        for g, sl in self._groups:
            w_g = weights[sl]
            second = (solved[sl] * w_g[:, None]).T @ solved[sl]  # sum_k w_k (B^-1 mu)(B^-1 mu)^T
            q = np.diag(alpha) - b_inverse[g]  # alpha_i alpha_j P_ij
            p = q * inv_alpha[:, None] * inv_alpha[None, :]
            hess += p * (0.5 * w_g.sum() * q + second)
        diagonal = v_bar - 0.5 + hp_hess
        if self.truncate:
            diagonal = diagonal - tr_hess
        hess[np.diag_indices(n)] += diagonal
        if outside.any():
            hess[outside, :] = 0.0
            hess[:, outside] = 0.0
            hess[outside, outside] = -1.0
        return Evaluation(value, grad, 0.5 * (hess + hess.T), log_evidence, weights)

    def __call__(self, theta: np.ndarray) -> float:
        return self.evaluate(theta, order=0).value

    def log_evidence(self, theta: np.ndarray) -> float:
        r"""Estimate of :math:`\log p(\mathcal{D}\mid\alpha)`, without the hyperprior."""
        return self.evaluate(theta, order=0).log_evidence

    def evidence_gain(self, theta: np.ndarray) -> np.ndarray:
        r"""Log-evidence lost by pruning each questionable parameter in turn.

        Entry ``i`` is :math:`\log\hat p(\mathcal{D}\mid\alpha) -
        \log\hat p(\mathcal{D}\mid\alpha_i\to\infty)`, the other precisions being
        held fixed: the log Bayes factor in favour of keeping the term.  It
        complements the relevance indicator, which measures how much the prior
        constrains a parameter but not how much the data ask for it.
        """
        theta = np.clip(np.asarray(theta, dtype=float), self.theta_lower, self.theta_upper)
        base = self.log_evidence(theta)
        gain = np.empty(self.n)
        for i in range(self.n):
            trial = theta.copy()
            trial[i] = self.theta_upper[i]
            gain[i] = base - self.log_evidence(trial)
        return gain

    # ------------------------------------------------------------------ #
    #  quantities at the optimum
    # ------------------------------------------------------------------ #
    def relevance(self, theta: np.ndarray, weighting: str = "posterior") -> np.ndarray:
        r"""Relevance indicator :math:`\gamma_i \in [0, 1]` of every questionable parameter.

        :math:`\gamma_{k,i} = 1 - \alpha_i P_{k,ii}` is the share of the posterior
        precision of :math:`\phi_i` that comes from the data rather than from the
        prior, according to kernel ``k`` (Eq. 22).  The kernels are combined by a
        root mean square, weighted either by their posterior weights
        (``"posterior"``, the default) or uniformly (``"uniform"``, the literal
        Eq. 23).  Both coincide whenever the kernels share one covariance, which is
        the case in every experiment of the paper.
        """
        theta = np.clip(np.asarray(theta, dtype=float), self.theta_lower, self.theta_upper)
        inv_alpha = np.exp(-theta)
        weights = self.evaluate(theta, order=0).weights
        n_kernels = self._mu.shape[0]
        total = np.zeros(self.n)
        for g, sl in self._groups:
            chol = cholesky(self._cov[g] + np.diag(inv_alpha), lower=True, check_finite=False)
            binv = cho_solve((chol, True), np.eye(self.n), check_finite=False)
            gamma = np.clip(np.diag(binv) * inv_alpha, 0.0, 1.0)
            if weighting == "posterior":
                share = weights[sl].sum()
            elif weighting == "uniform":
                share = (sl.stop - sl.start) / n_kernels
            else:
                raise ValueError(f"unknown weighting '{weighting}'")
            total += share * gamma**2
        return np.sqrt(total)

    def posterior(self, theta: np.ndarray) -> GaussianMixture:
        r"""Posterior :math:`\hat p(\phi\mid\mathcal{D},\alpha)` over all parameters (Eq. 12).

        The cross-covariance between questionable and known parameters is
        :math:`C_k - \Sigma_{\alpha,k} B_k^{-1} C_k`; Eq. (B.14) of the paper carries
        a spurious dependence on the kernel means.
        """
        theta = np.clip(np.asarray(theta, dtype=float), self.theta_lower, self.theta_upper)
        return self.mixture.multiply_gaussian(self.ard_index, 0.0, np.exp(-theta)).normalized()
