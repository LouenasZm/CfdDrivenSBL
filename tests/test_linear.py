"""NSBL reduces to classical sparse Bayesian learning on linear-Gaussian models (Section 2.7)."""

import numpy as np
import pytest

from sblcfd import ARDEvidence, GammaHyperprior, maximize_evidence
from sblcfd.linear import (
    linear_likelihood_mixture,
    log_evidence_linear,
    nsbl_linear,
    sbl_linear,
)


@pytest.fixture
def regression():
    rng = np.random.default_rng(0)
    n_data, n_coef = 60, 12
    design = rng.standard_normal((n_data, n_coef))
    coef = np.zeros(n_coef)
    coef[[1, 4, 7]] = [1.5, -2.0, 0.7]
    precision = 25.0
    data = design @ coef + rng.standard_normal(n_data) / np.sqrt(precision)
    return design, data, precision, coef


def test_likelihood_is_exactly_one_gaussian(regression):
    design, data, precision, _ = regression
    mixture = linear_likelihood_mixture(design, data, precision)
    rng = np.random.default_rng(1)
    phi = rng.standard_normal((5, design.shape[1]))
    residual = data[None, :] - phi @ design.T
    expected = 0.5 * data.size * np.log(precision / (2 * np.pi)) - 0.5 * precision * (
        residual**2
    ).sum(1)
    np.testing.assert_allclose(mixture.logpdf(phi), expected, rtol=1e-10)


def test_evidence_equals_closed_form(regression):
    design, data, precision, _ = regression
    objective = ARDEvidence(
        linear_likelihood_mixture(design, data, precision), hyperprior=GammaHyperprior.none()
    )
    rng = np.random.default_rng(2)
    for _ in range(5):
        alpha = np.exp(rng.uniform(-4, 4, design.shape[1]))
        assert objective.log_evidence(np.log(alpha)) == pytest.approx(
            log_evidence_linear(design, data, precision, alpha), abs=1e-9
        )


def test_sbl_fixed_point_is_a_stationary_point_of_nsbl(regression):
    design, data, precision, coef = regression
    reference = sbl_linear(design, data, precision=precision, tol=1e-12)
    kept = reference.relevance > 0.5
    np.testing.assert_array_equal(np.flatnonzero(kept), np.flatnonzero(coef))
    objective = ARDEvidence(
        linear_likelihood_mixture(design, data, precision), hyperprior=GammaHyperprior.none()
    )
    gradient = objective.evaluate(np.log(np.minimum(reference.alpha, 1e10)), 1).gradient
    assert np.max(np.abs(gradient)) < 1e-6
    # for the pruned terms the evidence still increases at a finite, large precision:
    # without a hyperprior their optimum is alpha = infinity
    theta = np.log(np.where(kept, reference.alpha, np.exp(6.0)))
    assert np.all(objective.evaluate(theta, 1).gradient[~kept] > 0.0)


def test_newton_and_fixed_point_agree_and_newton_is_cheaper(regression):
    design, data, precision, coef = regression
    fixed_point = sbl_linear(design, data, precision=precision)
    objective = ARDEvidence(
        linear_likelihood_mixture(design, data, precision), hyperprior=GammaHyperprior(1e-6, 1e-6)
    )
    newton = maximize_evidence(objective, n_starts=4, seed=0)[0]
    kept = np.flatnonzero(coef)
    np.testing.assert_allclose(newton.theta[kept], np.log(fixed_point.alpha[kept]), atol=1e-2)
    np.testing.assert_array_equal(newton.relevance > 0.5, fixed_point.relevance > 0.5)
    assert newton.n_iterations < fixed_point.n_iterations


def test_unknown_noise_is_recovered_by_both(regression):
    design, data, precision, coef = regression
    classic = sbl_linear(design, data)
    newton = nsbl_linear(design, data, hyperprior=GammaHyperprior(1e-5, 1e-5))
    for result in (classic, newton):
        np.testing.assert_array_equal(np.flatnonzero(result.relevance > 0.5), np.flatnonzero(coef))
        assert result.precision == pytest.approx(precision, rel=0.35)
        np.testing.assert_allclose(result.mean, coef, atol=0.15)
    np.testing.assert_allclose(newton.mean, classic.mean, atol=0.03)
    assert newton.n_evidence_evaluations < classic.n_evidence_evaluations


def test_underdetermined_problem_is_rejected():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        linear_likelihood_mixture(rng.standard_normal((5, 8)), rng.standard_normal(5), 1.0)
