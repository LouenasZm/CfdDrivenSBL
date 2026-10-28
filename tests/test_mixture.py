"""Gaussian-mixture container: densities, closed-form operations and constructors."""

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal as mvn

from sblcfd import GaussianMixture, fit_mixture


def random_cov(rng, d, scale=1.0):
    a = rng.standard_normal((d, d))
    return scale * (a @ a.T / d + 0.3 * np.eye(d))


def make(rng, structure, n_kernels=12, d=4):
    means = rng.standard_normal((n_kernels, d))
    logw = rng.standard_normal(n_kernels)
    if structure == "tied":
        return GaussianMixture(means, random_cov(rng, d, 0.3), logw)
    if structure == "grouped":
        covs = np.array([random_cov(rng, d, s) for s in (0.2, 0.6, 1.0)])
        return GaussianMixture(means, covs, logw, rng.integers(0, 3, n_kernels))
    return GaussianMixture(
        means, np.array([random_cov(rng, d, 0.4) for _ in range(n_kernels)]), logw
    )


def brute_logpdf(mix, x):
    terms = [
        mix.log_weights[k] + mvn.logpdf(x, mix.means[k], mix.covariances[mix.cov_index[k]])
        for k in range(mix.n_components)
    ]
    return logsumexp(np.array(terms), axis=0)


@pytest.mark.parametrize("structure", ["tied", "grouped", "full"])
def test_logpdf_matches_direct_sum(structure):
    rng = np.random.default_rng(0)
    mix = make(rng, structure)
    x = rng.standard_normal((25, mix.dim))
    np.testing.assert_allclose(mix.logpdf(x), brute_logpdf(mix, x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(mix.logpdf(x, normalized=True), brute_logpdf(mix, x) - mix.log_mass)


def test_kernels_are_sorted_by_covariance_and_weights_follow():
    rng = np.random.default_rng(1)
    means = rng.standard_normal((6, 2))
    covs = np.array([np.eye(2), 4.0 * np.eye(2)])
    index = np.array([1, 0, 1, 0, 0, 1])
    logw = np.log(np.arange(1.0, 7.0))
    mix = GaussianMixture(means, covs, logw, index)
    assert np.all(np.diff(mix.cov_index) >= 0)
    # every kernel keeps its own covariance and weight after the reordering
    for k in range(6):
        j = int(np.flatnonzero(np.all(mix.means == means[k], axis=1))[0])
        assert mix.cov_index[j] == index[k]
        assert mix.log_weights[j] == pytest.approx(logw[k])


@pytest.mark.parametrize("structure", ["tied", "grouped", "full"])
def test_moments_match_samples(structure):
    rng = np.random.default_rng(2)
    mix = make(rng, structure, d=3)
    draws = mix.sample(200_000, rng)
    np.testing.assert_allclose(draws.mean(axis=0), mix.mean(), atol=0.02)
    np.testing.assert_allclose(np.cov(draws.T), mix.covariance(), atol=0.03)
    np.testing.assert_allclose(mix.std(), np.sqrt(np.diag(mix.covariance())))


@pytest.mark.parametrize("structure", ["tied", "grouped", "full"])
def test_condition_is_a_slice_of_the_joint(structure):
    rng = np.random.default_rng(3)
    mix = make(rng, structure, d=4)
    index, rest = np.array([1, 3]), np.array([0, 2])
    value = np.array([0.3, -0.2])
    sliced = mix.condition(index, value)
    # mass of the slice = marginal density of the conditioned coordinates
    assert sliced.log_mass == pytest.approx(float(mix.marginal(index).logpdf(value[None])[0]))
    # density of the slice = joint density on the slice
    y = rng.standard_normal((10, 2))
    full = np.empty((10, 4))
    full[:, rest], full[:, index] = y, value
    np.testing.assert_allclose(sliced.logpdf(y), mix.logpdf(full), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("structure", ["tied", "grouped", "full"])
def test_multiply_gaussian_against_precision_form(structure):
    rng = np.random.default_rng(4)
    mix = make(rng, structure, d=4)
    index = np.array([0, 2])
    centre = np.array([0.4, -0.1])
    factor_cov = random_cov(rng, 2, 0.5)
    product = mix.multiply_gaussian(index, centre, factor_cov)

    embed = np.zeros((4, 2))
    embed[index, [0, 1]] = 1.0
    precision_factor = np.linalg.inv(factor_cov)
    for k in range(mix.n_components):
        sigma = mix.covariances[mix.cov_index[k]]
        post_cov = np.linalg.inv(np.linalg.inv(sigma) + embed @ precision_factor @ embed.T)
        post_mean = post_cov @ (
            np.linalg.solve(sigma, mix.means[k]) + embed @ precision_factor @ centre
        )
        np.testing.assert_allclose(product.means[k], post_mean, atol=1e-10)
        np.testing.assert_allclose(product.covariances[product.cov_index[k]], post_cov, atol=1e-10)
    # the product is a density identity: check it pointwise
    x = rng.standard_normal((8, 4))
    expected = mix.logpdf(x) + mvn.logpdf(x[:, index], centre, factor_cov)
    np.testing.assert_allclose(product.logpdf(x), expected, rtol=1e-9, atol=1e-9)


def test_multiply_gaussian_accepts_diagonal_and_scalar_mean():
    rng = np.random.default_rng(5)
    mix = make(rng, "tied", d=3)
    a = mix.multiply_gaussian([0, 1], 0.0, np.array([0.5, 2.0]))
    b = mix.multiply_gaussian([0, 1], np.zeros(2), np.diag([0.5, 2.0]))
    np.testing.assert_allclose(a.means, b.means)
    np.testing.assert_allclose(a.log_weights, b.log_weights)


def test_kde_reproduces_scipy_gaussian_kde():
    rng = np.random.default_rng(6)
    samples = rng.standard_normal((400, 3)) @ np.array(
        [[1.0, 0.4, 0.0], [0.0, 0.7, 0.2], [0, 0, 0.5]]
    )
    x = rng.standard_normal((20, 3))
    for rule in ("scott", "silverman"):
        mix = GaussianMixture.from_kde(samples, bandwidth=rule)
        reference = gaussian_kde(samples.T, bw_method=rule)
        np.testing.assert_allclose(mix.logpdf(x), reference.logpdf(x.T), rtol=1e-9, atol=1e-9)
    assert mix.is_tied and mix.n_components == 400


def test_kde_mass_weights_and_bandwidth_factor():
    rng = np.random.default_rng(7)
    samples = rng.standard_normal((300, 2))
    weights = rng.random(300)
    mix = GaussianMixture.from_kde(samples, weights, log_mass=3.5)
    assert mix.log_mass == pytest.approx(3.5)
    np.testing.assert_allclose(mix.weights, weights / weights.sum())
    narrow = GaussianMixture.from_kde(samples, factor=0.5)
    wide = GaussianMixture.from_kde(samples)
    np.testing.assert_allclose(narrow.covariances, 0.25 * wide.covariances)


def test_shrinkage_kde_preserves_first_two_moments():
    rng = np.random.default_rng(8)
    samples = rng.standard_normal((500, 4)) * np.array([1.0, 0.2, 3.0, 0.5]) + 2.0
    plain = GaussianMixture.from_kde(samples)
    shrunk = GaussianMixture.from_kde(samples, shrinkage=True)
    sample_cov = np.cov(samples.T)
    np.testing.assert_allclose(shrunk.mean(), samples.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(shrunk.covariance(), sample_cov, rtol=5e-3)
    # the plain estimate is inflated by 1 + h^2
    assert np.all(np.diag(plain.covariance()) > 1.2 * np.diag(sample_cov))


def test_single_gaussian_and_dispatch():
    rng = np.random.default_rng(9)
    samples = rng.standard_normal((200, 2))
    mix = fit_mixture(samples, "gaussian")
    assert mix.n_components == 1
    np.testing.assert_allclose(mix.mean(), samples.mean(axis=0))
    np.testing.assert_allclose(mix.covariance(), np.cov(samples.T))
    assert fit_mixture(samples, "kde").n_components == 200
    with pytest.raises(ValueError):
        fit_mixture(samples, "unknown")


def test_em_recovers_two_components_and_mirrors_selected_dims():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(10)
    a = rng.normal([-2.0, 0.0], [0.3, 0.5], size=(1500, 2))
    b = rng.normal([2.0, 1.0], [0.6, 0.2], size=(500, 2))
    mix = GaussianMixture.from_em(np.vstack([a, b]), n_components=(1, 2, 3), seed=0)
    assert mix.n_components == 2
    order = np.argsort(mix.means[:, 0])
    np.testing.assert_allclose(mix.means[order], [[-2.0, 0.0], [2.0, 1.0]], atol=0.1)
    np.testing.assert_allclose(mix.weights[order], [0.75, 0.25], atol=0.03)

    # samples piled up against zero: after mirroring one kernel centred at zero suffices
    piled = np.abs(rng.normal(0.0, 0.4, size=(4000, 1)))
    mirrored = GaussianMixture.from_em(piled, n_components=(1, 2), symmetric_dims=[0], seed=0)
    assert mirrored.n_components == 1
    assert abs(mirrored.mean()[0]) < 0.03
    assert mirrored.std()[0] == pytest.approx(0.4, rel=0.05)


def test_input_validation():
    with pytest.raises(ValueError):
        GaussianMixture(np.zeros((3, 2)), np.zeros((2, 2, 2)))  # 1 < G < K without index
    with pytest.raises(ValueError):
        GaussianMixture(np.zeros((3, 2)), np.eye(3))
    with pytest.raises(ValueError):
        GaussianMixture.from_kde(np.zeros((2, 5)))
