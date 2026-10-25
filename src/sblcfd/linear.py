r"""Linear-Gaussian reference: classical sparse Bayesian learning.

For a linear model :math:`d = \Psi\phi + \epsilon`, :math:`\epsilon\sim\mathcal{N}(0,\rho^{-1}I)`,
with ARD priors on every coefficient, NSBL reduces to the sparse Bayesian learning
(SBL) of Tipping (2001): the likelihood is *exactly* one Gaussian kernel in
:math:`\phi` (Section 2.7 of Sandhu et al., 2021).  This module provides

* that exact single-kernel mixture (:func:`linear_likelihood_mixture`),
* the classical SBL fixed-point iteration (:func:`sbl_linear`), used as an
  independent reference in the test-suite,
* the NSBL counterpart with noise re-estimation (:func:`nsbl_linear`).

It needs at least as many data as coefficients.  Under-determined problems
(compressive sensing, as in Bai, 2023) have no normalisable likelihood in
:math:`\phi` and are outside the scope of NSBL; use :func:`sbl_linear` there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .evidence import ARDEvidence, GammaHyperprior
from .mixture import LOG_2PI, GaussianMixture
from .optimize import maximize_evidence

__all__ = [
    "LinearSBLResult",
    "linear_likelihood_mixture",
    "log_evidence_linear",
    "nsbl_linear",
    "sbl_linear",
]


@dataclass
class LinearSBLResult:
    """Result of a linear sparse Bayesian regression.

    Attributes
    ----------
    alpha : ndarray
        ARD precisions.
    mean, covariance : ndarray
        Posterior moments of the coefficients.
    precision : float
        Noise precision :math:`\\rho`.
    relevance : ndarray
        :math:`\\gamma_i = 1 - \\alpha_i P_{ii}`.
    log_evidence : float
    n_iterations : int
        Fixed-point sweeps (SBL) or outer noise updates (NSBL).
    n_evidence_evaluations : int
        Evidence evaluations spent, the cost measure of Fig. 2b of the paper.
    """

    alpha: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    precision: float
    relevance: np.ndarray
    log_evidence: float
    n_iterations: int
    n_evidence_evaluations: int = 0


def linear_likelihood_mixture(
    design: np.ndarray, data: np.ndarray, precision: float
) -> GaussianMixture:
    r"""Exact Gaussian representation of the likelihood of a linear model (Eq. 26).

    :math:`\mathcal{N}(d\mid\Psi\phi,\rho^{-1}I) = a\,\mathcal{N}(\phi\mid\mu,\Sigma)` with
    :math:`\mu` the least-squares estimate and :math:`\Sigma = (\rho\Psi^T\Psi)^{-1}`.
    """
    design = np.asarray(design, dtype=float)
    data = np.asarray(data, dtype=float).ravel()
    n_data, n_coef = design.shape
    if n_data < n_coef:
        raise ValueError("the likelihood of an under-determined model is not normalisable")
    gram = design.T @ design
    factor = cho_factor(gram, lower=True)
    mu = cho_solve(factor, design.T @ data)
    cov = cho_solve(factor, np.eye(n_coef)) / precision
    residual = data - design @ mu
    logdet_cov = -2.0 * np.log(np.diag(factor[0])).sum() - n_coef * np.log(precision)
    log_amplitude = (
        0.5 * n_data * (np.log(precision) - LOG_2PI)
        - 0.5 * precision * float(residual @ residual)
        + 0.5 * (n_coef * LOG_2PI + logdet_cov)
    )
    return GaussianMixture.from_gaussian(mu, cov, log_amplitude)


def log_evidence_linear(
    design: np.ndarray, data: np.ndarray, precision: float, alpha: np.ndarray
) -> float:
    r"""Closed-form :math:`\log\mathcal{N}(d\mid 0,\ \Psi A^{-1}\Psi^T + \rho^{-1}I)`."""
    design = np.asarray(design, dtype=float)
    data = np.asarray(data, dtype=float).ravel()
    cov = (design / np.asarray(alpha, dtype=float)) @ design.T
    cov[np.diag_indices_from(cov)] += 1.0 / precision
    factor = cho_factor(cov, lower=True)
    quad = float(data @ cho_solve(factor, data))
    logdet = 2.0 * np.log(np.diag(factor[0])).sum()
    return -0.5 * (data.size * LOG_2PI + logdet + quad)


def _posterior(design, data, precision, alpha):
    cov = np.linalg.inv(precision * design.T @ design + np.diag(alpha))
    mean = precision * cov @ (design.T @ data)
    return mean, cov


def sbl_linear(
    design: np.ndarray,
    data: np.ndarray,
    *,
    precision: float | None = None,
    hyperprior: GammaHyperprior | None = None,
    noise_shape: float = 0.0,
    noise_rate: float = 0.0,
    max_iterations: int = 5000,
    tol: float = 1e-6,
    alpha_max: float = 1e12,
) -> LinearSBLResult:
    r"""Classical SBL by fixed-point re-estimation (Tipping, 2001; Eq. 28 of the paper).

    Each sweep sets :math:`\alpha_i \leftarrow (\gamma_i + 2r_i)/(m_i^2 + 2s_i)` and, if
    ``precision`` is ``None``, re-estimates the noise precision.
    """
    design = np.asarray(design, dtype=float)
    data = np.asarray(data, dtype=float).ravel()
    n_data, n_coef = design.shape
    hyperprior = GammaHyperprior.none() if hyperprior is None else hyperprior
    estimate_noise = precision is None
    rho = 1.0 / np.var(data) if estimate_noise else float(precision)
    alpha = np.ones(n_coef)

    n_sweeps = 0
    for _ in range(max_iterations):
        n_sweeps += 1
        mean, cov = _posterior(design, data, rho, alpha)
        gamma = 1.0 - alpha * np.diag(cov)
        new_alpha = (gamma + 2.0 * hyperprior.shape) / (mean**2 + 2.0 * hyperprior.rate)
        new_alpha = np.clip(new_alpha, 1e-12, alpha_max)
        if estimate_noise:
            residual = data - design @ mean
            rho = (n_data - gamma.sum() + 2.0 * noise_shape) / (
                float(residual @ residual) + 2.0 * noise_rate
            )
        change = np.max(np.abs(np.log(new_alpha) - np.log(alpha)))
        alpha = new_alpha
        if change < tol:
            break

    mean, cov = _posterior(design, data, rho, alpha)
    return LinearSBLResult(
        alpha=alpha,
        mean=mean,
        covariance=cov,
        precision=float(rho),
        relevance=1.0 - alpha * np.diag(cov),
        log_evidence=log_evidence_linear(design, data, rho, alpha),
        n_iterations=n_sweeps,
        n_evidence_evaluations=n_sweeps,
    )


def nsbl_linear(
    design: np.ndarray,
    data: np.ndarray,
    *,
    precision: float | None = None,
    hyperprior: GammaHyperprior | None = None,
    noise_shape: float = 0.0,
    noise_rate: float = 0.0,
    theta0: float | np.ndarray = 5.0,
    max_outer: int = 100,
    tol: float = 1e-6,
) -> LinearSBLResult:
    r"""NSBL on a linear model: Newton iterations on *all* precisions at once.

    The noise precision, when unknown, is re-estimated after every Newton solve as
    in SBL, :math:`\rho^{-1} = (\lVert d-\Psi m\rVert^2 + 2b)/(N_d - \sum_i\gamma_i + 2a)`,
    and the Newton iteration is warm-started from the previous precisions.

    Parameters
    ----------
    theta0 : float or array_like
        Initial :math:`\log\alpha`.  The default starts from an empty model, as in
        Section 3.1 of the paper.
    """
    design = np.asarray(design, dtype=float)
    data = np.asarray(data, dtype=float).ravel()
    n_data, n_coef = design.shape
    hyperprior = GammaHyperprior() if hyperprior is None else hyperprior
    estimate_noise = precision is None
    rho = 1.0 / np.var(data) if estimate_noise else float(precision)
    theta = np.broadcast_to(np.asarray(theta0, dtype=float), (n_coef,)).copy()

    n_evaluations = n_outer = 0
    for _ in range(max_outer):
        n_outer += 1
        objective = ARDEvidence(linear_likelihood_mixture(design, data, rho), hyperprior=hyperprior)
        # evidence_tol=0: pure type-II maximum likelihood, the rule classical SBL applies
        best = maximize_evidence(objective, starts=theta[None, :], evidence_tol=0.0)[0]
        n_evaluations += best.n_iterations
        theta = best.theta
        if not estimate_noise:
            break
        alpha = np.exp(theta)
        mean, cov = _posterior(design, data, rho, alpha)
        gamma = 1.0 - alpha * np.diag(cov)
        residual = data - design @ mean
        new_rho = (n_data - gamma.sum() + 2.0 * noise_shape) / (
            float(residual @ residual) + 2.0 * noise_rate
        )
        converged = abs(np.log(new_rho) - np.log(rho)) < tol
        rho = float(new_rho)
        if converged:
            break

    alpha = np.exp(theta)
    mean, cov = _posterior(design, data, rho, alpha)
    return LinearSBLResult(
        alpha=alpha,
        mean=mean,
        covariance=cov,
        precision=float(rho),
        relevance=1.0 - alpha * np.diag(cov),
        log_evidence=log_evidence_linear(design, data, rho, alpha),
        n_iterations=n_outer,
        n_evidence_evaluations=n_evaluations,
    )
