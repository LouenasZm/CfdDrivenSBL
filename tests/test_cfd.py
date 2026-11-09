"""CFD-driven layer: Kriging, POD emulators, field likelihoods, dictionaries and adapters."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import qmc

from sblcfd import NSBL, model_average
from sblcfd.cfd import (
    FieldLikelihood,
    FieldMisfit,
    KrigingPredictor,
    PODEmulator,
    effective_sample_size,
    fit_kriging,
    likelihood_from_bilevel_data,
    masked_rows,
    monomial,
    toy_cases,
    transition_library,
)
from sblcfd.cfd.kriging import _negative_log_likelihood, _trend_basis


# --------------------------------------------------------------------------- #
#  Kriging
# --------------------------------------------------------------------------- #
def smooth_function(x):
    return np.c_[np.sin(2 * x[:, 0]) + x[:, 1] ** 2, np.cos(x[:, 2]) * x[:, 0], x[:, 1] * x[:, 2]]


@pytest.fixture(scope="module")
def design():
    x = qmc.Sobol(3, seed=0).random(128) * 2.0 - 1.0
    return x, smooth_function(x)


@pytest.mark.parametrize("trend", ["constant", "linear"])
def test_likelihood_gradient_matches_finite_differences(design, trend):
    x, y = design
    xn = (x - x.mean(0)) / x.std(0)
    yn = (y - y.mean(0)) / y.std(0)
    d2 = (xn[:, None, :] - xn[None, :, :]) ** 2
    basis = _trend_basis(xn, trend)
    p = np.array([-0.6, -1.0, -0.3])
    value, grad, _ = _negative_log_likelihood(p, d2, basis, yn, 1e-10)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1e-6
        up = _negative_log_likelihood(p + e, d2, basis, yn, 1e-10)[0]
        down = _negative_log_likelihood(p - e, d2, basis, yn, 1e-10)[0]
        assert grad[i] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-4)
    assert np.isfinite(value)


@pytest.mark.parametrize("shared", [True, False])
def test_kriging_interpolates_and_predicts(design, shared):
    x, y = design
    model = fit_kriging(x, y, shared=shared, seed=0)
    assert model.n_outputs == 3 and model.dim == 3 and model.n_train == 128
    # interpolation, up to the nugget that keeps the correlation matrix well conditioned
    np.testing.assert_allclose(model.predict(x), y, atol=1e-3)
    x_test = np.random.default_rng(1).uniform(-1, 1, (200, 3))
    error = model.predict(x_test) - smooth_function(x_test)
    assert np.sqrt(np.mean(error**2)) < 2e-2
    # chunked evaluation gives the same numbers
    np.testing.assert_allclose(model.predict(x_test, chunk=1000), model.predict(x_test))
    lower, upper = model.bounds()
    np.testing.assert_allclose(lower, x.min(0))
    np.testing.assert_allclose(upper, x.max(0))


def smt_like(model: KrigingPredictor, k: int, trend: str = "constant", pls: bool = False):
    """Object exposing the attributes read from a trained SMT Kriging model."""
    eta = model.eta[:, k]
    fake = SimpleNamespace(
        options={"corr": "squar_exp", "poly": trend},
        X_norma=model.x_train,
        X_offset=model.x_offset,
        X_scale=model.x_scale,
        optimal_theta=eta,
        optimal_par={"beta": model.beta[:, k : k + 1], "gamma": model.gamma[:, k : k + 1]},
        y_mean=np.array([model.y_mean[k]]),
        y_std=np.array([model.y_std[k]]),
    )
    if pls:  # KPLS stores one theta per component and the weights of the components
        weights = np.sqrt(np.c_[0.25 * eta, 0.75 * eta])
        fake.optimal_theta = np.array([1.0, 1.0])
        fake.coeff_pls = weights
    return fake


@pytest.mark.parametrize("pls", [False, True])
def test_predictor_rebuilt_from_smt_attributes(design, pls):
    x, y = design
    model = fit_kriging(x, y, shared=False, trend="linear", seed=0)
    rebuilt = KrigingPredictor.from_smt([smt_like(model, k, "linear", pls) for k in range(3)])
    x_test = np.random.default_rng(2).uniform(-1, 1, (50, 3))
    np.testing.assert_allclose(
        rebuilt.predict(x_test), model.predict(x_test), rtol=1e-10, atol=1e-12
    )

    wrong = smt_like(model, 0)
    wrong.options["corr"] = "matern52"
    with pytest.raises(NotImplementedError):
        KrigingPredictor.from_smt([wrong])
    shifted = smt_like(model, 1)
    shifted.X_norma = model.x_train + 1.0
    with pytest.raises(ValueError):
        KrigingPredictor.from_smt([smt_like(model, 0), shifted])


# --------------------------------------------------------------------------- #
#  POD emulator and field misfit
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def field_problem():
    """Fields that depend smoothly on three inputs, living on a 20 x 12 grid."""
    rng = np.random.default_rng(0)
    x = qmc.Sobol(3, seed=1).random(128)
    gx, gy = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 12), indexing="ij")

    def solver(inputs):
        inputs = np.atleast_2d(inputs)
        a, b, c = inputs[:, 0, None, None], inputs[:, 1, None, None], inputs[:, 2, None, None]
        field = np.tanh(4 * (gx - 0.3 - 0.4 * a)) + b * np.sin(3 * gy) + c**2 * gx * gy
        return field.reshape(inputs.shape[0], -1)

    emulator = PODEmulator.fit(x, solver(x), energy=0.99999, seed=0)
    return rng, solver, emulator, (20, 12)


def test_pod_emulator_accuracy_and_shapes(field_problem):
    rng, solver, emulator, shape = field_problem
    x_test = rng.random((60, 3))
    truth = solver(x_test)
    prediction = emulator.predict(x_test)
    assert prediction.shape == truth.shape == (60, shape[0] * shape[1])
    q2 = 1 - ((prediction - truth) ** 2).sum() / ((truth - truth.mean(0)) ** 2).sum()
    assert q2 > 0.999
    assert emulator.energy > 0.9999
    np.testing.assert_allclose(
        emulator.modes.T @ emulator.modes, np.eye(emulator.n_modes), atol=1e-8
    )
    rows = np.array([3, 17, 100])
    np.testing.assert_allclose(emulator.predict(x_test, rows), prediction[:, rows])
    assert emulator.validate(solver, x_test) < 0.02
    assert emulator.validate(lambda q: solver(q).T, x_test, rows) < 0.02  # (n_points, n) layout
    assert PODEmulator.fit(x_test, truth, n_modes=2, seed=0).n_modes == 2


def test_reduced_misfit_equals_full_space_misfit(field_problem):
    rng, solver, emulator, shape = field_problem
    reference = solver(np.array([[0.4, 0.6, 0.2]]))[0] + 0.01 * rng.standard_normal(
        shape[0] * shape[1]
    )
    rows = np.flatnonzero(rng.random(reference.size) < 0.6)
    weights = rng.random(rows.size) + 0.5
    x = rng.random((40, 3))

    for w in (None, weights):
        reduced = FieldMisfit(emulator, reference, rows=rows, weights=w)
        generic = FieldMisfit(emulator.predict, reference, rows=rows, weights=w)
        normalised = np.ones(rows.size) if w is None else w * rows.size / w.sum()
        direct = ((emulator.predict(x, rows) - reference[rows]) ** 2) @ normalised
        np.testing.assert_allclose(reduced.sse(x), direct, rtol=1e-9)
        np.testing.assert_allclose(generic.sse(x), direct, rtol=1e-9)
        np.testing.assert_allclose(reduced.mse(x), direct / rows.size, rtol=1e-9)
    # reference restricted beforehand, and the residual helpers
    restricted = FieldMisfit(emulator, reference[rows], rows=rows)
    np.testing.assert_allclose(
        restricted.sse(x), FieldMisfit(emulator, reference, rows=rows).sse(x)
    )
    residual = restricted.residual(x[0])
    assert residual @ residual == pytest.approx(restricted.sse(x[:1])[0])
    laid_out = restricted.residual_field(x[0], shape)
    assert laid_out.shape == shape and np.isnan(laid_out).sum() == reference.size - rows.size

    with pytest.raises(ValueError):
        FieldMisfit(emulator, np.full(reference.size, np.nan))
    with pytest.raises(ValueError):
        FieldMisfit(emulator, reference, n_eff=0.0)


def test_likelihood_noise_models(field_problem):
    rng, solver, emulator, _ = field_problem
    reference = solver(np.array([[0.4, 0.6, 0.2]]))[0]
    terms = [
        FieldMisfit(emulator, reference + 0.02, name="a", n_eff=30.0, scale=0.5),
        FieldMisfit(emulator, 2 * reference, name="b", scale=2.0),
    ]
    x = rng.random((25, 3))
    sse = np.column_stack([t.sse(x) for t in terms])
    n = np.array([t.n_points for t in terms])
    n_eff = np.array([30.0, n[1]])

    marginal = FieldLikelihood(terms)
    expected = (
        gammaln(n_eff / 2)
        - 0.5 * n_eff * np.log(2 * np.pi)
        - 0.5 * n_eff * np.log(0.5 * n_eff / n * sse)
    ).sum(1)
    np.testing.assert_allclose(marginal(x), expected, rtol=1e-10)
    # scale invariance: multiplying a field by a constant only shifts the log-likelihood
    scaled = FieldLikelihood(
        [terms[0], FieldMisfit(lambda q: 10 * emulator.predict(q), 20 * reference, n_eff=n[1])]
    )
    difference = scaled(x) - marginal(x)
    np.testing.assert_allclose(difference, difference[0], rtol=1e-9)

    sigma = np.array([0.1, 0.3])
    fixed = FieldLikelihood(terms, noise="fixed", sigma=sigma)
    expected = (
        -n_eff * np.log(sigma * np.sqrt(2 * np.pi)) - 0.5 * (n_eff / n) * sse / sigma**2
    ).sum(1)
    np.testing.assert_allclose(fixed(x), expected, rtol=1e-10)

    np.testing.assert_allclose(marginal.mse(x), sse / n)
    np.testing.assert_allclose(
        marginal.normalised_loss(x), (sse / n / np.array([0.5, 2.0])).mean(1)
    )
    np.testing.assert_allclose(marginal.improvement(x), 1 - sse / n / np.array([0.5, 2.0]))
    np.testing.assert_allclose(marginal.noise_level(x), np.sqrt(sse / n))
    assert marginal.names == ["a", "b"]

    # n_eff is read at call time, so it can be refined between two runs
    before = marginal(x)
    terms[1].n_eff = 10.0
    assert not np.allclose(marginal(x), before)
    with pytest.raises(ValueError):
        FieldLikelihood(terms, noise="fixed")
    with pytest.raises(ValueError):
        FieldLikelihood([FieldMisfit(emulator, reference)]).normalised_loss(x)


def test_effective_sample_size():
    rng = np.random.default_rng(0)
    white = rng.standard_normal((60, 40))
    assert effective_sample_size(white) > 0.8 * white.size
    # a field made of 6 x 4 constant blocks carries about 24 independent values
    blocks = np.kron(rng.standard_normal((6, 4)), np.ones((10, 10)))
    assert 12 < effective_sample_size(blocks) < 60
    # a uniform bias is a single piece of information
    assert effective_sample_size(np.full((30, 30), 0.7)) < 3.0
    masked = white.copy()
    masked[:, 20:] = np.nan
    assert 0.8 * 1200 < effective_sample_size(masked) <= 1200
    assert effective_sample_size(rng.standard_normal(500)) > 400
    with pytest.raises(ValueError):
        effective_sample_size(np.full((4, 4), np.nan))


# --------------------------------------------------------------------------- #
#  symbolic dictionary
# --------------------------------------------------------------------------- #
def test_transition_library_layout_and_evaluation():
    library = transition_library()
    assert library.n_parameters == 21
    assert library.names[0] == "RP[1]" and library.names[11] == "RE[1]"
    assert library.kinds.count("constant") == 2 and library.kinds.count("exponent") == 13
    parameters = library.parameters()
    assert [(p.lower, p.upper) for p in parameters[:5]] == [
        (0, 1),
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (0, 1),
    ]

    coefficients = np.zeros(21)
    coefficients[[0, 1, 4, 11, 15]] = [0.2, -0.3, 0.5, 0.6, 0.7]
    features = {"lam": 0.5, "eta1": np.array([0.0, 1.0]), "eta2": 0.3, "eta3": 2.0}
    values = library.evaluate(coefficients, features)
    np.testing.assert_allclose(values["RP"], (0.2 - 0.15) * np.exp(-0.5 * np.array([0.0, 1.0])))
    np.testing.assert_allclose(values["RE"], 0.6 * np.exp(-0.7 * np.array([0.0, 2.0])))
    # a batch of coefficient vectors broadcasts against the features
    batch = library.evaluate(np.stack([coefficients, 2 * coefficients])[:, None, :], features)
    assert batch["RP"].shape == (2, 2)

    text = library.format(coefficients)
    assert text.splitlines()[0] == "RP = (0.2 - 0.3 lam) * exp(-[0.5 eta1])"
    assert text.splitlines()[1] == "RE = 0.6 * exp(-[0.7 eta1*eta3])"
    assert library.format(np.zeros(21)) == "RP = 0\nRE = 0"
    assert "±" in library.format(coefficients, std=np.full(21, 0.01))
    assert monomial(eta1=2, eta2=1).label(latex=True) == "eta1^{2} eta2"
    assert monomial().label() == "1"


def test_toy_model_is_a_coupled_nonlinear_map():
    cases = toy_cases(n_x=32, n_y=12)
    assert set(cases) == {"zpg-low", "zpg-high", "pg-mixed", "pg-adverse", "pg-favourable"}
    baseline, truth = np.zeros(21), np.zeros(21)
    truth[[0, 1, 4, 11, 13, 15]] = [0.25, -0.35, 0.9, 0.5, 0.4, 0.8]
    for name, case in cases.items():
        gamma = case.intermittency(np.stack([baseline, truth]))
        assert gamma.shape == (2, 32)
        assert np.all((gamma > 0) & (gamma <= 1)) and gamma[0, 0] < 0.1 and gamma[0, -1] > 0.9
        fields = case.fields(np.stack([baseline, truth]))
        assert fields["u"].shape == fields["dudy"].shape == (2, 32 * 12)
        assert case.skin_friction(truth).shape == (1, 32)
        assert not np.allclose(gamma[0], gamma[1]), name
        mask = case.boundary_layer_mask(baseline)
        assert mask.dtype == bool and 0 < mask.sum() < mask.size
    # the pressure-gradient terms act only where there is a pressure gradient
    probe = baseline.copy()
    probe[1] = 0.5
    np.testing.assert_allclose(
        cases["zpg-low"].intermittency(probe), cases["zpg-low"].intermittency(baseline)
    )
    assert not np.allclose(
        cases["pg-mixed"].intermittency(probe), cases["pg-mixed"].intermittency(baseline)
    )
    with pytest.raises(ValueError):
        cases["zpg-low"].intermittency(np.zeros(5))


# --------------------------------------------------------------------------- #
#  adapter to the deterministic framework
# --------------------------------------------------------------------------- #
class FakeSurrogate:
    """Mimics the pickled surrogates: ``.model`` is an SMT PODI, ``.predict`` is (n_points, n)."""

    def __init__(self, emulator: PODEmulator, rebuildable: bool = True):
        self._emulator = emulator
        if rebuildable:
            predictor = emulator.predictor
            self.model = SimpleNamespace(
                basis=emulator.modes,
                mean=emulator.mean[:, None],
                interp_coeff=[smt_like(predictor, k) for k in range(predictor.n_outputs)],
            )
        else:
            self.n_inputs = 3

    def predict(self, x):
        return self._emulator.predict(x).T


def reference_loss(x, cases, variables, data, threshold):
    """The loss of ``bilevel.loss.compute_normalised_loss``, written out independently."""
    total = np.zeros(x.shape[0])
    for case in cases:
        s_star = data["les"][case]["S_star"].ravel()
        for variable in variables:
            prediction = data["surrogates"][case][variable].predict(x)
            reference = data["les"][case][variable].ravel()
            finite = np.isfinite(reference)
            keep = s_star[finite] < threshold
            error = prediction[finite][keep] - reference[finite][keep, None]
            total += (error**2).mean(axis=0) / data["baseline"]["loss"][case][variable]
    return total / (len(cases) * len(variables))


@pytest.fixture(scope="module")
def bilevel_data(field_problem):
    rng, solver, emulator, shape = field_problem
    data = {"surrogates": {}, "les": {}, "baseline": {"loss": {}}}
    for case, shift in (("t3a", 0.0), ("t3b", 0.1)):
        data["surrogates"][case], data["les"][case], data["baseline"]["loss"][case] = {}, {}, {}
        for variable, factor in (("uu", 1.0), ("duy", 0.5)):
            reference = factor * solver(np.array([[0.4 + shift, 0.6, 0.2]]))[0]
            reference = reference + 0.01 * rng.standard_normal(reference.size)
            reference[rng.integers(0, reference.size, 7)] = np.nan
            data["les"][case][variable] = reference.reshape(shape)
            data["surrogates"][case][variable] = FakeSurrogate(
                PODEmulator(factor * emulator.mean, factor * emulator.modes, emulator.predictor)
            )
            data["baseline"]["loss"][case][variable] = 0.3 + rng.random()
        data["les"][case]["S_star"] = rng.uniform(0.0, 1.2, shape)
    return data


def test_masked_rows_follows_the_deterministic_loss():
    reference = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
    s_star = np.array([0.1, 0.2, 0.9, 0.3, 0.1, 0.5])
    rows, values = masked_rows(reference, s_star, 0.8, 6)  # surrogate on the full field
    np.testing.assert_array_equal(rows, [0, 3, 5])
    np.testing.assert_allclose(values, [1.0, 4.0, 6.0])
    rows, values = masked_rows(reference, s_star, 0.8, 4)  # surrogate on the finite points only
    np.testing.assert_array_equal(rows, [0, 2, 3])
    np.testing.assert_allclose(values, [1.0, 4.0, 6.0])
    with pytest.raises(ValueError):
        masked_rows(reference, s_star, 0.8, 5)


def test_adapter_reproduces_the_deterministic_loss(bilevel_data):
    cases, variables = ["t3a", "t3b"], ["uu", "duy"]
    likelihood = likelihood_from_bilevel_data(bilevel_data, cases, variables, s_star_threshold=0.8)
    assert likelihood.names == ["t3a/uu", "t3a/duy", "t3b/uu", "t3b/duy"]
    assert all(term._reduced for term in likelihood.terms)  # the fast path was taken
    x = np.random.default_rng(3).random((12, 3))
    np.testing.assert_allclose(
        likelihood.normalised_loss(x),
        reference_loss(x, cases, variables, bilevel_data, 0.8),
        rtol=1e-8,
    )
    assert np.all(np.isfinite(likelihood(x)))
    # residuals can be laid out on the LES grid, to estimate the effective sample size
    term = likelihood.terms[0]
    shape = bilevel_data["les"]["t3a"]["uu"].shape
    laid_out = term.residual_field(x[0], shape)
    kept = np.isfinite(bilevel_data["les"]["t3a"]["uu"]) & (
        bilevel_data["les"]["t3a"]["S_star"] < 0.8
    )
    np.testing.assert_array_equal(np.isfinite(laid_out), kept)
    assert 1.0 <= effective_sample_size(laid_out) <= term.n_points
    # per-field effective sample sizes, and the leader layout of the baseline
    leader = dict(bilevel_data, baseline=bilevel_data["baseline"]["loss"])
    n_eff = {c: {v: 25.0 for v in variables} for c in cases}
    tempered = likelihood_from_bilevel_data(leader, cases, variables, n_eff=n_eff)
    assert all(term.n_eff == 25.0 for term in tempered.terms)


def test_adapter_falls_back_to_the_original_surrogate(bilevel_data):
    cases, variables = ["t3a"], ["uu"]
    original = bilevel_data["surrogates"]["t3a"]["uu"]
    x = np.random.default_rng(4).random((6, 3))
    expected = reference_loss(x, cases, variables, bilevel_data, 0.8)

    # (a) a surrogate that is not an SMT PODI at all
    plain = dict(bilevel_data, surrogates={"t3a": {"uu": FakeSurrogate(original._emulator, False)}})
    likelihood = likelihood_from_bilevel_data(plain, cases, variables)
    assert not likelihood.terms[0]._reduced
    np.testing.assert_allclose(likelihood.normalised_loss(x), expected, rtol=1e-8)

    # (b) a PODI whose internals do not reproduce its own predictions
    broken = FakeSurrogate(original._emulator)
    broken.model.mean = broken.model.mean + 1.0
    inconsistent = dict(bilevel_data, surrogates={"t3a": {"uu": broken}})
    with pytest.warns(UserWarning, match="deviates"):
        likelihood = likelihood_from_bilevel_data(inconsistent, cases, variables)
    assert not likelihood.terms[0]._reduced
    np.testing.assert_allclose(likelihood.normalised_loss(x), expected, rtol=1e-8)


# --------------------------------------------------------------------------- #
#  end to end on the toy transition problem (reduced size)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_sparse_correction_of_the_toy_transition_model():
    library = transition_library()
    parameters = library.parameters()
    lower = np.array([p.lower for p in parameters])
    upper = np.array([p.upper for p in parameters])
    cases = toy_cases(n_x=40, n_y=16)
    truth = np.zeros(21)
    truth[[0, 1, 4, 11, 13, 15]] = [0.25, -0.35, 0.9, 0.5, 0.4, 0.8]
    rng = np.random.default_rng(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        x = lower + (upper - lower) * qmc.Sobol(21, seed=0).random(256)
    terms = []
    for name in ("zpg-low", "zpg-high", "pg-mixed"):
        case = cases[name]
        snapshots, reference = case.fields(x), case.fields(truth)
        mask = case.boundary_layer_mask(truth)
        for quantity in ("u", "dudy"):
            emulator = PODEmulator.fit(x, snapshots[quantity], energy=0.9999, seed=0)
            data = reference[quantity][0] + 2e-3 * rng.standard_normal(mask.size)
            term = FieldMisfit(emulator, data, rows=np.flatnonzero(mask), name=f"{name}/{quantity}")
            term.scale = float(term.mse(np.zeros(21))[0])
            term.n_eff = effective_sample_size(term.residual_field(np.zeros(21), case.shape))
            terms.append(term)
    likelihood = FieldLikelihood(terms)
    assert likelihood.normalised_loss(np.zeros((1, 21)))[0] == pytest.approx(1.0)

    model = NSBL(likelihood, parameters, truncate_prior=True)
    model.sample(1500, seed=0, max_steps=20)
    mixture = model.fit_mixture(seed=0)
    assert mixture.n_components <= 8  # "auto" picks the EM mixture in 21 dimensions
    result = model.optimize(n_starts=12, seed=0)

    # the two constants carry the bulk of the correction and are never pruned
    assert result.active[0] and result.active[11]
    assert result.evidence_gain[0] > 5.0
    assert 2 <= result.active.sum() <= 12  # a sparse model out of 21 candidate terms
    probability = result.inclusion_probability()
    assert probability[0] == pytest.approx(1.0) and np.all((probability >= 0) & (probability <= 1))

    ranked = model.compare_structures(n_candidates=2, n_particles=800, seed=1, max_steps=20)
    assert ranked[0].log_evidence >= ranked[-1].log_evidence
    best = ranked[0]
    assert np.all(best.samples[:, ~best.active] == 0.0)
    assert (
        likelihood.normalised_loss(best.mean()[None])[0] < 0.3
    )  # at least 70 % below the baseline

    # a posteriori: through the solver itself, on a case never seen in training
    test_case = cases["pg-adverse"]
    exact = test_case.skin_friction(truth)[0]
    baseline_error = np.abs(test_case.skin_friction(np.zeros(21))[0] - exact).mean()
    draws = model_average(ranked, 100, seed=0)
    assert draws.shape == (100, 21)
    assert np.abs(test_case.skin_friction(draws).mean(axis=0) - exact).mean() < 0.5 * baseline_error
