"""Sequential Monte Carlo sampler: moments, evidence, bounds and multimodality."""

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import multivariate_normal as mvn

from sblcfd import HalfNormal, JointPrior, Normal, SMCSampler, Uniform


def test_reflect_returns_points_inside_and_parity():
    prior = JointPrior([Uniform(0.0, 1.0), Normal(0.0, 1.0), HalfNormal(1.0)])
    x = np.array([[-0.2, 5.0, -0.3], [1.3, -5.0, 0.4], [2.4, 0.0, 0.1], [0.5, 0.0, 0.2]])
    reflected, flipped = prior.reflect(x)
    np.testing.assert_allclose(
        reflected, [[0.2, 5.0, 0.3], [0.7, -5.0, 0.4], [0.4, 0.0, 0.1], [0.5, 0.0, 0.2]]
    )
    # 2.4 is reflected twice (at 1 and at 0): even parity, hence not flagged
    np.testing.assert_array_equal(
        flipped,
        [[True, False, True], [True, False, False], [False, False, False], [False, False, False]],
    )
    assert np.all(np.isfinite(prior.logpdf(reflected)))


def gaussian_problem(d, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    cov = 0.02 * (a @ a.T / d + 0.2 * np.eye(d))
    mean = rng.uniform(-0.3, 0.3, d)
    whitening = np.linalg.inv(np.linalg.cholesky(cov))
    logdet = np.linalg.slogdet(cov)[1]

    def loglik(x):
        z = (x - mean) @ whitening.T
        return -0.5 * (z * z).sum(axis=1) - 0.5 * (d * np.log(2 * np.pi) + logdet)

    return loglik, mean, cov


@pytest.mark.parametrize("independent_moves", [True, False])
def test_gaussian_target(independent_moves):
    d = 6
    loglik, mean, cov = gaussian_problem(d)
    result = SMCSampler(1500, seed=0, independent_moves=independent_moves).run(
        loglik, JointPrior([Uniform(-2, 2)] * d)
    )
    sd = np.sqrt(np.diag(cov))
    assert np.max(np.abs(result.samples.mean(axis=0) - mean) / sd) < 0.15
    np.testing.assert_allclose(result.samples.std(axis=0) / sd, 1.0, atol=0.1)
    # normalised likelihood, uniform prior on a box of volume 4^d
    assert result.log_evidence == pytest.approx(-d * np.log(4.0), abs=0.35)
    assert result.betas[0] == 0.0 and result.betas[-1] == 1.0
    assert result.n_stages == len(result.acceptance) == len(result.correlation)
    np.testing.assert_allclose(result.log_likelihood, loglik(result.samples))


def test_bimodal_target_keeps_both_modes():
    def loglik(x):
        a = mvn.logpdf(x, [-1.5, 0.0], 0.05 * np.eye(2)) + np.log(0.7)
        b = mvn.logpdf(x, [1.5, 0.5], 0.05 * np.eye(2)) + np.log(0.3)
        return np.logaddexp(a, b)

    result = SMCSampler(2000, seed=0).run(loglik, JointPrior([Uniform(-4, 4)] * 2))
    assert np.mean(result.samples[:, 0] < 0) == pytest.approx(0.7, abs=0.06)
    assert result.log_evidence == pytest.approx(-np.log(64.0), abs=0.2)


def test_correlated_target_in_a_corner():
    """Strongly correlated likelihood piled against two bounds.

    This is the configuration in which a reflected random walk needs its Hastings
    correction: without it the evidence and the moments are biased.
    """
    cov = np.array([[0.04, 0.036], [0.036, 0.04]])
    precision = np.linalg.inv(cov)
    centre = np.array([0.05, -0.05])

    def loglik(x):
        d = x - centre
        return -0.5 * np.einsum("ni,ij,nj->n", d, precision, d)

    grid = np.linspace(0.0, 1.0, 801)
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    density = np.exp(loglik(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
    exact = np.log(trapezoid(trapezoid(density, grid, axis=1), grid))
    p = density / density.sum()
    mean = np.array([(p * gx).sum(), (p * gy).sum()])
    sd = np.sqrt(np.array([(p * (gx - mean[0]) ** 2).sum(), (p * (gy - mean[1]) ** 2).sum()]))

    errors, means = [], []
    for seed in range(6):
        result = SMCSampler(2000, seed=seed, independent_moves=False).run(
            loglik, JointPrior([Uniform(0, 1)] * 2)
        )
        assert result.samples.min() >= 0.0 and result.samples.max() <= 1.0
        errors.append(result.log_evidence - exact)
        means.append(result.samples.mean(axis=0))
    assert abs(np.mean(errors)) < 0.04
    np.testing.assert_allclose(np.mean(means, axis=0), mean, atol=0.06 * sd.max())


def test_error_indicator_and_bookkeeping():
    loglik, _, _ = gaussian_problem(3)
    result = SMCSampler(800, seed=3).run(loglik, JointPrior([Uniform(-2, 2)] * 3))
    assert result.log_evidence_error >= 0.0
    assert result.n_evaluations >= 800
    assert result.samples.shape == (800, 3)
    assert result.wall_time > 0.0


def test_invalid_likelihoods_are_reported():
    prior = JointPrior([Uniform(0, 1)] * 2)
    with pytest.raises(RuntimeError):
        SMCSampler(100, seed=0).run(lambda x: np.full(x.shape[0], -np.inf), prior)
    with pytest.raises(ValueError):
        SMCSampler(100, seed=0).run(lambda x: np.zeros(3), prior)
    with pytest.raises(ValueError):
        SMCSampler(100, ess_fraction=1.5)


def test_non_finite_values_count_as_zero_likelihood():
    def loglik(x):
        value = -0.5 * ((x[:, 0] - 0.5) / 0.1) ** 2
        return np.where(x[:, 0] > 0.8, np.nan, value)  # e.g. a diverged simulation

    result = SMCSampler(1000, seed=0).run(loglik, JointPrior([Uniform(0, 1)]))
    assert result.samples.max() <= 0.8
    assert result.samples.mean() == pytest.approx(0.5, abs=0.02)
