"""End-to-end NSBL workflow, including the examples of Sandhu et al. (2021)."""

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import norm

from sblcfd import (
    NSBL,
    GammaHyperprior,
    GaussianMixture,
    GaussianMixturePrior,
    Parameter,
    Uniform,
    questionable,
)
from sblcfd.linear import linear_likelihood_mixture


# --------------------------------------------------------------------------- #
#  parameters
# --------------------------------------------------------------------------- #
def test_parameter_validation():
    assert Parameter("a", -1.0, 1.0).questionable
    known = Parameter("k", prior=Uniform(0.0, 5.0))
    assert not known.questionable and (known.lower, known.upper) == (0.0, 5.0)
    with pytest.raises(ValueError):
        Parameter("a", 0.5, 1.0)  # zero is not admissible for a prunable parameter
    with pytest.raises(ValueError):
        Parameter("a", 1.0, -1.0)
    names = [p.name for p in questionable(["x", "y"], [0.0, -1.0], 1.0)]
    assert names == ["x", "y"]
    with pytest.raises(ValueError):
        NSBL(lambda x: x[:, 0], [Parameter("k", prior=Uniform(0, 1))])  # nothing to sparsify
    with pytest.raises(ValueError):
        NSBL(lambda x: x[:, 0], [Parameter("a", -1, 1), Parameter("a", -1, 1)])


# --------------------------------------------------------------------------- #
#  Section 3.2: polynomial regression with a multimodal prior
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def polynomial():
    rng = np.random.default_rng(3)
    x = np.linspace(0.75, 1.25, 50)
    y = 1.0 + x**2 + rng.normal(0.0, np.sqrt(0.02), 50)
    design = np.c_[np.ones(50), x, x**2]

    def loglik(phi):
        residual = y[None, :] - phi @ design.T
        return -0.5 * (residual**2).sum(axis=1) / 0.02 - 25 * np.log(2 * np.pi * 0.02)

    parameters = [
        Parameter("a0", prior=GaussianMixturePrior([-1.0, 0.0, 1.0], 0.2)),
        Parameter("a1", -10.0, 10.0),
        Parameter("a2", -10.0, 10.0),
    ]
    likelihood = linear_likelihood_mixture(design, y, 1 / 0.02)
    parts = [likelihood.multiply_gaussian([0], c, np.array([0.04])) for c in (-1.0, 0.0, 1.0)]
    exact = GaussianMixture(
        np.vstack([p.means for p in parts]),
        np.vstack([p.covariances for p in parts]),
        np.concatenate([p.log_weights for p in parts]) - np.log(3.0),
    )
    return loglik, parameters, exact


def test_multimodal_evidence_has_the_two_optima_of_the_paper(polynomial):
    loglik, parameters, exact = polynomial
    model = NSBL(loglik, parameters)
    model.set_mixture(exact)

    local = model.optimize(starts=np.array([[6.0, 8.0]]))  # y = 2x: a1 kept, a2 pruned
    np.testing.assert_array_equal(local.relevant, [True, False])
    best = model.optimize(starts=np.array([[-3.0, -3.0]]))  # y = 1 + x^2: a1 pruned, a2 kept
    np.testing.assert_array_equal(best.relevant, [False, True])
    assert best.objective > local.objective

    multistart = model.optimize(n_starts=12, seed=0)
    assert len(multistart.optima) == 2
    np.testing.assert_array_equal(multistart.relevant, [False, True])
    np.testing.assert_array_equal(multistart.active, [True, False, True])
    mean, std = multistart.moments()
    np.testing.assert_allclose(mean, [1.0, 0.0, 1.0], atol=0.15)
    assert std[1] < 0.2  # the pruned coefficient is pinned near zero
    assert "pruned" in multistart.summary() and "known" in multistart.summary()
    assert multistart.to_dict()["active"] == [True, False, True]


def test_sampled_workflow_agrees_with_the_exact_mixture(polynomial):
    loglik, parameters, exact = polynomial
    model = NSBL(loglik, parameters)
    sampling = model.sample(2500, seed=1)
    # mass of likelihood x known prior: sampled estimate vs analytic value
    assert model.log_mass_ == pytest.approx(exact.log_mass, abs=0.1)
    weights = [
        np.mean(sampling.samples[:, 0] < -0.5),
        np.mean(np.abs(sampling.samples[:, 0]) < 0.5),
        np.mean(sampling.samples[:, 0] > 0.5),
    ]
    np.testing.assert_allclose(weights, exact.weights, atol=0.05)

    model.fit_mixture("kde")
    sampled = model.optimize(n_starts=12, seed=0)
    model.set_mixture(exact)
    reference = model.optimize(n_starts=12, seed=0)
    np.testing.assert_array_equal(sampled.relevant, reference.relevant)
    assert sampled.log_evidence == pytest.approx(reference.log_evidence, abs=0.6)
    np.testing.assert_allclose(sampled.log_alpha[1], reference.log_alpha[1], atol=0.3)


def test_convenience_fit_and_external_samples(polynomial):
    loglik, parameters, _ = polynomial
    result = NSBL(loglik, parameters).fit(1500, seed=2, n_starts=8)
    np.testing.assert_array_equal(result.relevant, [False, True])

    model = NSBL(loglik, parameters)
    with pytest.raises(RuntimeError):
        model.fit_mixture()
    with pytest.raises(RuntimeError):
        model.optimize()
    with pytest.raises(ValueError):
        model.set_samples(np.zeros((10, 2)))
    model.set_samples(NSBL(loglik, parameters).sample(1500, seed=2).samples)
    model.fit_mixture("gaussian")
    assert model.optimize(n_starts=4, seed=0).posterior.n_components == 1


# --------------------------------------------------------------------------- #
#  sign-constrained parameters: half-normal / truncated ARD priors
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def constrained():
    """y = a x1 + b x2 + noise with a, b in [0, 2]; a = 0.8 is needed, b = 0 is not."""
    rng = np.random.default_rng(5)
    x1, x2 = rng.standard_normal(30), rng.standard_normal(30)
    y = 0.8 * x1 + rng.normal(0.0, 0.3, 30)

    def loglik(phi):
        residual = y[None, :] - phi[:, :1] * x1[None, :] - phi[:, 1:] * x2[None, :]
        return -0.5 * (residual**2).sum(axis=1) / 0.09 - 30 * np.log(0.3 * np.sqrt(2 * np.pi))

    grid = np.linspace(0.0, 2.0, 601)
    ga, gb = np.meshgrid(grid, grid, indexing="ij")
    likelihood = np.exp(loglik(np.c_[ga.ravel(), gb.ravel()])).reshape(ga.shape)

    def exact_log_evidence(theta, truncated):
        sd = np.exp(-0.5 * np.asarray(theta))
        prior = np.ones_like(ga)
        for values, s in zip((ga, gb), sd):
            mass = norm.cdf(2.0 / s) - 0.5 if truncated else 0.5
            prior = prior * norm.pdf(values, 0.0, s) / mass
        return float(np.log(trapezoid(trapezoid(likelihood * prior, grid, axis=1), grid)))

    return loglik, questionable(["a", "b"], 0.0, 2.0), exact_log_evidence


@pytest.mark.parametrize("truncate", [False, True])
def test_bounded_evidence_matches_quadrature(constrained, truncate):
    loglik, parameters, exact_log_evidence = constrained
    model = NSBL(loglik, parameters, truncate_prior=truncate, hyperprior=GammaHyperprior.none())
    np.testing.assert_array_equal(model.fold_index, [0, 1])
    model.sample(4000, seed=0)
    model.fit_mixture("kde")
    objective = model.evidence()
    offset = 0.0 if truncate else 2 * np.log(2.0)  # two half-normal priors
    for theta in ([0.0, 0.0], [0.5, 4.0], [-1.0, 2.0], [3.0, 3.0]):
        estimate = objective.log_evidence(np.array(theta)) + offset
        assert estimate == pytest.approx(exact_log_evidence(theta, truncate), abs=0.15)


@pytest.mark.parametrize("truncate", [False, True])
def test_sign_constrained_selection_and_posterior(constrained, truncate):
    loglik, parameters, _ = constrained
    model = NSBL(loglik, parameters, truncate_prior=truncate)
    model.sample(3000, seed=1)
    model.fit_mixture("kde")
    result = model.optimize(n_starts=8, seed=0)
    np.testing.assert_array_equal(result.relevant, [True, False])
    draws = result.sample(5000, seed=0)
    assert draws.min() >= 0.0 and draws.max() <= 2.0  # folded and inside the bounds
    mean, _ = result.moments()
    assert mean[0] == pytest.approx(0.8, abs=0.15)
    assert mean[1] < 0.1


def test_relevance_is_invariant_to_the_units_of_a_parameter(constrained):
    """Rescaling a coefficient must not change which terms are kept.

    Here ``b`` is expressed in units 1000 times larger, so its samples are 1000
    times smaller.  A Gamma hyperprior with a positive rate, applied literally,
    then caps its precision far below what pruning requires; applied to the
    dimensionless precision it does not care, and neither does the flat default.
    """
    loglik, _, _ = constrained

    def rescaled(phi):
        return loglik(phi * np.array([1.0, 1e3]))

    parameters = [Parameter("a", 0.0, 2.0), Parameter("b", 0.0, 2e-3)]

    def relevance(likelihood, params, **options):
        model = NSBL(likelihood, params, **options)
        model.sample(3000, seed=1)
        model.fit_mixture("kde")
        return model.optimize(n_starts=8, seed=0).relevance

    reference = relevance(loglik, questionable(["a", "b"], 0.0, 2.0))
    assert reference[0] > 0.9 and reference[1] < 0.05
    np.testing.assert_allclose(relevance(rescaled, parameters), reference, atol=0.02)

    paper = GammaHyperprior.paper()
    scaled = relevance(rescaled, parameters, hyperprior=paper, standardize=True)
    literal = relevance(rescaled, parameters, hyperprior=paper, standardize=False)
    assert scaled[0] > 0.9
    assert scaled[1] < 0.5 < literal[1]  # the unscaled rate prevents the pruning
    assert literal[1] > 0.9


def test_em_mixture_mirrors_coefficients_piled_at_zero(constrained):
    pytest.importorskip("sklearn")
    loglik, parameters, exact_log_evidence = constrained
    model = NSBL(loglik, parameters, truncate_prior=True, hyperprior=GammaHyperprior.none())
    model.sample(4000, seed=0)
    mixture = model.fit_mixture("em", seed=0)
    assert mixture.n_components <= 8
    result = model.optimize(n_starts=8, seed=0)
    np.testing.assert_array_equal(result.relevant, [True, False])
    theta = np.array([0.5, 4.0])
    assert model.evidence().log_evidence(theta) == pytest.approx(
        exact_log_evidence(theta, True), abs=0.2
    )


def test_calibration_of_the_sparse_model(constrained):
    loglik, parameters, _ = constrained
    model = NSBL(loglik, parameters, truncate_prior=True)
    full = model.sample(2000, seed=1)
    model.fit_mixture("kde")
    result = model.optimize(n_starts=8, seed=0)
    calibration = model.calibrate(n_particles=2000, seed=2)
    np.testing.assert_array_equal(calibration.active, [True, False])
    assert np.all(calibration.samples[:, 1] == 0.0)
    assert calibration.mean()[0] == pytest.approx(0.8, abs=0.15)
    assert calibration.best()[1] == 0.0
    assert calibration.quantile([0.025, 0.975]).shape == (2, 2)
    assert "active" in calibration.summary()
    # the sparse structure is preferred over the full one
    assert calibration.log_evidence > full.log_evidence
    # slicing the mixture posterior gives a comparable answer for free
    sliced, n_effective = result.condition_on_pruned()
    assert sliced.dim == 1 and n_effective > 30
    assert sliced.mean()[0] == pytest.approx(calibration.mean()[0], abs=0.1)
    with pytest.raises(ValueError):
        model.calibrate(active=[True])
