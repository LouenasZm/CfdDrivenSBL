"""Semi-analytical evidence: derivatives, exactness, stability and the bounded-prior variants."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm

from sblcfd import ARDEvidence, GammaHyperprior, GaussianMixture
from sblcfd.evidence import truncation_normaliser


def random_cov(rng, d, scale=1.0):
    a = rng.standard_normal((d, d))
    return scale * (a @ a.T / d + 0.3 * np.eye(d))


def make(rng, structure, n_kernels=30, d=5):
    means = rng.standard_normal((n_kernels, d))
    logw = rng.standard_normal(n_kernels)
    if structure == "tied":
        return GaussianMixture(means, random_cov(rng, d, 0.2), logw)
    if structure == "grouped":
        covs = np.array([random_cov(rng, d, s) for s in (0.1, 0.5, 1.0)])
        return GaussianMixture(means, covs, logw, rng.integers(0, 3, n_kernels))
    return GaussianMixture(
        means, np.array([random_cov(rng, d, 0.3) for _ in range(n_kernels)]), logw
    )


def finite_differences(objective, theta, h=1e-5):
    n = theta.size
    grad, hess = np.zeros(n), np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        grad[i] = (objective(theta + e) - objective(theta - e)) / (2 * h)
        hess[i] = (
            objective.evaluate(theta + e, 1).gradient - objective.evaluate(theta - e, 1).gradient
        ) / (2 * h)
    return grad, hess


@pytest.mark.parametrize("structure", ["tied", "grouped", "full"])
@pytest.mark.parametrize("subset", [None, [0, 2, 4]])
@pytest.mark.parametrize("truncate", [False, True])
def test_gradient_and_hessian_match_finite_differences(structure, subset, truncate):
    rng = np.random.default_rng(11)
    mix = make(rng, structure)
    n = mix.dim if subset is None else len(subset)
    objective = ARDEvidence(
        mix,
        subset,
        hyperprior=GammaHyperprior(0.3, 0.05),
        lower=np.r_[np.zeros(n // 2), -2.0 * np.ones(n - n // 2)],
        upper=np.r_[np.ones(n // 2), np.full(n - n // 2, np.inf)],
        truncate=truncate,
    )
    theta = rng.uniform(-3, 3, n)
    exact = objective.evaluate(theta, 2)
    grad, hess = finite_differences(objective, theta)
    np.testing.assert_allclose(exact.gradient, grad, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(exact.hessian, hess, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(exact.hessian, exact.hessian.T)
    # the three orders are consistent with one another
    assert objective.evaluate(theta, 0).value == pytest.approx(exact.value)
    np.testing.assert_allclose(objective.evaluate(theta, 1).gradient, exact.gradient)


@pytest.mark.parametrize("structure", ["tied", "grouped", "full"])
def test_evidence_and_posterior_against_brute_force(structure):
    """Kernel by kernel: precision-form posterior, and Bayes' identity for the evidence."""
    rng = np.random.default_rng(12)
    mix = make(rng, structure, n_kernels=10, d=5)
    index = np.array([0, 2, 4])
    theta = rng.uniform(-2, 2, 3)
    alpha = np.exp(theta)
    objective = ARDEvidence(mix, index, hyperprior=GammaHyperprior.none())
    posterior = objective.posterior(theta)

    prior_precision = np.zeros((5, 5))
    prior_precision[index, index] = alpha
    point = rng.standard_normal(5)
    log_terms = []
    for k in range(mix.n_components):
        sigma = mix.covariances[mix.cov_index[k]]
        cov = np.linalg.inv(np.linalg.inv(sigma) + prior_precision)
        mean = cov @ np.linalg.solve(sigma, mix.means[k])
        np.testing.assert_allclose(posterior.means[k], mean, atol=1e-10)
        np.testing.assert_allclose(posterior.covariances[posterior.cov_index[k]], cov, atol=1e-10)
        log_terms.append(
            mix.log_weights[k]
            + mvn.logpdf(point, mix.means[k], sigma)
            + mvn.logpdf(point[index], np.zeros(3), np.diag(1 / alpha))
            - mvn.logpdf(point, mean, cov)
        )
    log_terms = np.array(log_terms)
    assert objective.log_evidence(theta) == pytest.approx(logsumexp(log_terms), abs=1e-10)
    np.testing.assert_allclose(
        posterior.weights, np.exp(log_terms - logsumexp(log_terms)), atol=1e-10
    )
    assert posterior.log_mass == pytest.approx(0.0, abs=1e-12)


def test_extreme_precisions_stay_finite():
    rng = np.random.default_rng(13)
    objective = ARDEvidence(make(rng, "tied", d=6))
    for theta in (
        np.full(6, 30.0),
        np.full(6, -30.0),
        np.array([30.0, -30.0, 12.0, -12.0, 25.0, 0.0]),
        np.full(6, 60.0),  # beyond theta_max: the barrier takes over
    ):
        out = objective.evaluate(theta, 2)
        assert np.isfinite(out.value)
        assert np.all(np.isfinite(out.gradient)) and np.all(np.isfinite(out.hessian))
        gamma = objective.relevance(theta)
        assert np.all((gamma >= 0.0) & (gamma <= 1.0))
    # pruned parameters have zero relevance, free ones full relevance
    gamma = objective.relevance(np.array([30.0, -30.0, 30.0, -30.0, 30.0, -30.0]))
    np.testing.assert_allclose(gamma, [0, 1, 0, 1, 0, 1], atol=1e-8)


def test_barrier_pushes_back_inside():
    rng = np.random.default_rng(14)
    objective = ARDEvidence(make(rng, "tied", d=3), theta_max=10.0)
    out = objective.evaluate(np.array([12.0, 0.0, -13.0]), 2)
    assert out.gradient[0] == pytest.approx(-2.0)
    assert out.gradient[2] == pytest.approx(3.0)
    assert out.hessian[0, 0] == pytest.approx(-1.0) and out.hessian[0, 1] == 0.0


def test_relevance_weightings_agree_for_shared_covariance():
    rng = np.random.default_rng(15)
    tied = ARDEvidence(make(rng, "tied"))
    theta = rng.uniform(-2, 2, tied.n)
    np.testing.assert_allclose(
        tied.relevance(theta, "posterior"), tied.relevance(theta, "uniform"), atol=1e-12
    )
    full = ARDEvidence(make(rng, "full"))
    assert not np.allclose(full.relevance(theta, "posterior"), full.relevance(theta, "uniform"))
    with pytest.raises(ValueError):
        full.relevance(theta, "nonsense")


def test_truncation_normaliser_values_and_derivatives():
    theta = np.array([-8.0, -1.0, 0.5, 3.0])
    lower = np.array([0.0, -1.0, -np.inf, -0.3])
    upper = np.array([1.0, 1.0, 2.0, np.inf])
    log_z, g1, g2 = truncation_normaliser(theta, lower, upper)
    sd = np.exp(-0.5 * theta)
    np.testing.assert_allclose(
        np.exp(log_z), norm.cdf(upper / sd) - norm.cdf(lower / sd), rtol=1e-12
    )
    h = 1e-5
    up, down = (
        truncation_normaliser(theta + h, lower, upper),
        truncation_normaliser(theta - h, lower, upper),
    )
    np.testing.assert_allclose(g1, (up[0] - down[0]) / (2 * h), atol=1e-8)
    np.testing.assert_allclose(g2, (up[1] - down[1]) / (2 * h), atol=1e-8)
    # wide prior: the mass grows like sqrt(alpha); narrow prior: all (or half) the mass is inside
    assert truncation_normaliser(np.array([-25.0]), np.array([-1.0]), np.array([1.0]))[1][
        0
    ] == pytest.approx(0.5)
    assert np.exp(
        truncation_normaliser(np.array([25.0]), np.array([-1.0]), np.array([1.0]))[0][0]
    ) == pytest.approx(1.0)
    assert np.exp(
        truncation_normaliser(np.array([25.0]), np.array([0.0]), np.array([1.0]))[0][0]
    ) == pytest.approx(0.5)


def test_truncated_prior_tends_to_the_uniform_prior_evidence():
    """alpha -> 0: a truncated Gaussian becomes uniform, so the evidence tends to mass / volume."""
    rng = np.random.default_rng(16)
    means = rng.uniform(0.3, 0.7, (40, 2))
    mix = GaussianMixture(means, 1e-4 * np.eye(2), np.full(40, np.log(3.0 / 40)))
    lower, upper = np.array([0.0, -1.0]), np.array([1.0, 2.0])
    objective = ARDEvidence(
        mix, lower=lower, upper=upper, truncate=True, hyperprior=GammaHyperprior.none()
    )
    volume = np.prod(upper - lower)
    assert objective.log_evidence(np.full(2, -20.0)) == pytest.approx(
        np.log(3.0 / volume), abs=1e-4
    )
    with pytest.raises(ValueError):
        ARDEvidence(mix, lower=np.array([0.1, -1.0]), upper=upper, truncate=True)


def test_half_normal_prior_equals_formula_on_mirrored_kernels():
    """Sign-constrained parameter: reflected kernels + half-normal prior == plain formula x 2."""
    rng = np.random.default_rng(17)
    centres = np.abs(rng.normal(0.0, 0.5, 60))
    bandwidth = 0.2
    mix = GaussianMixture(centres[:, None], np.array([[bandwidth**2]]), np.full(60, -np.log(60)))
    objective = ARDEvidence(mix, hyperprior=GammaHyperprior.none())

    def reflected_density(x):
        return np.mean(norm.pdf(x, centres, bandwidth) + norm.pdf(-x, centres, bandwidth))

    def integrand(x, sd):
        return reflected_density(x) * 2 * norm.pdf(x, 0, sd)

    for theta in (-2.0, 0.0, 1.5, 4.0):
        integral, _ = quad(integrand, 0, np.inf, args=(np.exp(-0.5 * theta),), limit=200)
        assert objective.log_evidence(np.array([theta])) + np.log(2) == pytest.approx(
            np.log(integral), abs=1e-7
        )


def test_hyperprior_terms():
    prior = GammaHyperprior(shape=np.array([0.5, 1.0]), rate=np.array([0.1, 0.2]))
    theta = np.array([0.3, -1.0])
    value, grad, hess = prior.terms(theta, np.exp(theta))
    assert value == pytest.approx(np.sum(prior.shape * theta - prior.rate * np.exp(theta)))
    np.testing.assert_allclose(grad, prior.shape - prior.rate * np.exp(theta))
    np.testing.assert_allclose(hess, -prior.rate * np.exp(theta))
    assert GammaHyperprior.none().terms(theta, np.exp(theta))[0] == 0.0
    with pytest.raises(ValueError):
        GammaHyperprior(-1.0, 1.0)
