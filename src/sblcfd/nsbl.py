r"""Nonlinear sparse Bayesian learning: the three-stage workflow.

1. **Sample** ``likelihood x known prior`` once (:meth:`NSBL.sample`).
2. **Approximate** it with a Gaussian mixture (:meth:`NSBL.fit_mixture`).
3. **Optimise** the ARD precisions with Newton's method on the semi-analytical
   evidence (:meth:`NSBL.optimize`).

Stage 1 is the only one that touches the likelihood, and it never has to be
repeated: changing the hyperprior, the mixture or the optimiser reuses the samples.
A fourth, optional stage (:meth:`NSBL.calibrate`) re-samples the selected sparse
model, which gives its posterior and its evidence free of any mixture
approximation.

Sign-constrained parameters
---------------------------
A questionable parameter restricted to :math:`[0, u]` cannot carry a Gaussian ARD
prior.  It carries the half-normal one, :math:`2\,\mathcal{N}(\phi\mid0,\alpha^{-1})`
on :math:`\phi\ge0`.  Because that prior is even, the evidence under it equals the
evidence of the *mirror-symmetric extension* of the likelihood under the plain
Gaussian prior, and for a kernel mixture the mirrored kernels integrate to exactly
the same value as the original ones.  The NSBL formulas therefore apply unchanged;
only the posterior has to be folded back, :math:`\phi_i \leftarrow |\phi_i|`.
See ``docs/theory.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .evidence import ARDEvidence, GammaHyperprior
from .mixture import GaussianMixture
from .optimize import Optimum, maximize_evidence
from .priors import JointPrior, Prior, Uniform
from .smc import SMCResult, SMCSampler

__all__ = ["NSBL", "Calibration", "NSBLResult", "Parameter", "model_average", "questionable"]


@dataclass(frozen=True)
class Parameter:
    """One model parameter.

    Parameters
    ----------
    name : str
    lower, upper : float
        Admissible range.  For a questionable parameter it must contain zero (the
        value a pruned parameter takes) and be finite if the built-in sampler is
        used; in CFD-driven applications it is the range covered by the surrogate.
    prior : Prior, optional
        Known prior of an *a priori relevant* parameter.  ``None`` marks the
        parameter as *questionable*: it receives an ARD prior whose precision is
        learnt from the data.
    """

    name: str
    lower: float = -np.inf
    upper: float = np.inf
    prior: Prior | None = None

    def __post_init__(self) -> None:
        if self.prior is not None:
            object.__setattr__(self, "lower", max(float(self.lower), float(self.prior.lower)))
            object.__setattr__(self, "upper", min(float(self.upper), float(self.prior.upper)))
        if not self.upper > self.lower:
            raise ValueError(f"parameter '{self.name}': upper must exceed lower")
        if self.prior is None and not (self.lower <= 0.0 <= self.upper):
            raise ValueError(
                f"parameter '{self.name}' is questionable, so zero must be admissible; "
                f"got [{self.lower}, {self.upper}]. Shift the parameter or give it a known prior."
            )

    @property
    def questionable(self) -> bool:
        """Whether the parameter carries an ARD prior."""
        return self.prior is None


def questionable(
    names: Sequence[str], lower: Sequence[float] | float, upper: Sequence[float] | float
) -> list[Parameter]:
    """Build a list of questionable parameters from names and bounds."""
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (len(names),))
    upper = np.broadcast_to(np.asarray(upper, dtype=float), (len(names),))
    return [Parameter(n, float(lo), float(hi)) for n, lo, hi in zip(names, lower, upper)]


def _fold(x: np.ndarray, lower: np.ndarray, upper: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Mirror the coordinates in ``index`` about zero onto their admissible side."""
    if index.size:
        sign = np.where(lower[index] == 0.0, 1.0, -1.0)
        x[:, index] = sign * np.abs(x[:, index])
    return x


@dataclass
class NSBLResult:
    """Outcome of the evidence maximisation.

    Attributes
    ----------
    parameters : list of Parameter
    ard_index : ndarray of int
        Positions of the questionable parameters.
    log_alpha : ndarray
        Optimal :math:`\\log\\alpha` of every questionable parameter.
    relevance : ndarray
        Relevance indicator :math:`\\gamma_i \\in [0, 1]` of every questionable
        parameter.
    relevance_tol : float
        A questionable parameter is kept when its indicator exceeds this value.
    evidence_gain : ndarray
        Log-evidence lost when each questionable parameter is pruned on its own,
        the others being kept as they are: the log Bayes factor in favour of the
        term.  Zero for the pruned ones.
    log_evidence : float
        Log-evidence of the selected optimum.
    objective : float
        Log-evidence plus log-hyperprior.
    posterior : GaussianMixture
        Posterior over all parameters, before folding and truncation; use
        :meth:`sample` for draws that respect the bounds.
    optima : list of Optimum
        All distinct local optima, best first.  Each is a candidate sparse model.
    """

    parameters: list[Parameter]
    ard_index: np.ndarray
    log_alpha: np.ndarray
    relevance: np.ndarray
    relevance_tol: float
    evidence_gain: np.ndarray
    log_evidence: float
    objective: float
    posterior: GaussianMixture
    optima: list[Optimum]
    _moments: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def alpha(self) -> np.ndarray:
        """Optimal ARD precisions."""
        return np.exp(self.log_alpha)

    @property
    def relevant(self) -> np.ndarray:
        """Boolean mask over the questionable parameters: ``True`` if kept."""
        return self.relevance > self.relevance_tol

    @property
    def active(self) -> np.ndarray:
        """Boolean mask over *all* parameters: known ones and relevant questionable ones."""
        mask = np.ones(len(self.parameters), dtype=bool)
        mask[self.ard_index] = self.relevant
        return mask

    @property
    def _bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower = np.array([p.lower for p in self.parameters])
        upper = np.array([p.upper for p in self.parameters])
        fold = np.array(
            [i for i in self.ard_index if lower[i] == 0.0 or upper[i] == 0.0], dtype=int
        )
        return lower, upper, fold

    def sample(self, n: int, seed: int | np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` points from the posterior, folded and restricted to the bounds."""
        rng = np.random.default_rng(seed)
        lower, upper, fold = self._bounds
        collected: list[np.ndarray] = []
        missing = n
        for _ in range(200):
            x = _fold(self.posterior.sample(max(2 * missing, 256), rng), lower, upper, fold)
            x = x[np.all((x >= lower) & (x <= upper), axis=1)]
            collected.append(x[:missing])
            missing -= collected[-1].shape[0]
            if missing <= 0:
                break
        else:
            raise RuntimeError(
                "the posterior has almost no mass inside the bounds; the mixture "
                "approximation is probably too coarse near a boundary"
            )
        return np.vstack(collected)

    def moments(self, n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of every parameter (bounds respected)."""
        if self._moments is None:
            draws = self.sample(n, seed)
            self._moments = (draws.mean(axis=0), draws.std(axis=0))
        return self._moments

    def structures(self, n: int | None = None) -> list[np.ndarray]:
        """Distinct sparse structures among the local optima, best first.

        Each structure is a boolean mask over all parameters (known ones are always
        active).  With a collinear dictionary several structures explain the data
        almost equally well; :meth:`NSBL.compare_structures` ranks them without the
        mixture approximation.
        """
        found: list[np.ndarray] = []
        for optimum in self.optima:
            mask = np.ones(len(self.parameters), dtype=bool)
            mask[self.ard_index] = optimum.relevance > self.relevance_tol
            if not any(np.array_equal(mask, other) for other in found):
                found.append(mask)
        return found if n is None else found[:n]

    def inclusion_probability(self) -> np.ndarray:
        """Evidence-weighted frequency with which each questionable parameter is kept.

        The local optima are weighted by the exponential of their objective.  A
        value near one means every competitive sparse model contains the term; an
        intermediate value flags a term that can be traded against others.
        """
        objective = np.array([o.objective for o in self.optima])
        weights = np.exp(objective - objective.max())
        weights /= weights.sum()
        kept = np.array([o.relevance > self.relevance_tol for o in self.optima], dtype=float)
        return weights @ kept

    def condition_on_pruned(self) -> tuple[GaussianMixture, float]:
        """Posterior of the kept parameters with the pruned ones set exactly to zero.

        Obtained by slicing the posterior mixture, so it costs nothing.  The second
        return value is the effective number of kernels supporting the slice; when
        it is small (say below 30) the slice rests on too few samples and
        :meth:`NSBL.calibrate` should be used instead.
        """
        pruned = np.flatnonzero(~self.active)
        if pruned.size == 0:
            return self.posterior, float(1.0 / np.sum(self.posterior.weights**2))
        sliced = self.posterior.condition(pruned, 0.0).normalized()
        return sliced, float(1.0 / np.sum(sliced.weights**2))

    def summary(self) -> str:
        """Human-readable table of the result."""
        mean, std = self.moments()
        width = max(max(len(p.name) for p in self.parameters), 9)
        lines = [
            f"NSBL: log-evidence = {self.log_evidence:.4f}; "
            f"{int(self.relevant.sum())} of {self.ard_index.size} questionable parameters kept "
            f"(relevance > {self.relevance_tol:g}); {len(self.optima)} distinct optima",
            f"  {'parameter':<{width}}  {'log(alpha)':>10}  {'relevance':>9}  {'gain':>8}"
            f"  {'status':<6}  {'mean':>11}  {'std':>10}",
        ]
        position = {int(i): k for k, i in enumerate(self.ard_index)}
        for i, p in enumerate(self.parameters):
            if i in position:
                k = position[i]
                status = "kept" if self.relevant[k] else "pruned"
                head = (
                    f"{self.log_alpha[k]:>10.3f}  {self.relevance[k]:>9.3f}"
                    f"  {self.evidence_gain[k]:>8.2f}  {status:<6}"
                )
            else:
                head = f"{'-':>10}  {'-':>9}  {'-':>8}  {'known':<6}"
            lines.append(f"  {p.name:<{width}}  {head}  {mean[i]:>11.4g}  {std[i]:>10.3g}")
        lines.append("  (gain: log-evidence lost if that term alone is pruned)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable summary."""
        mean, std = self.moments()
        return {
            "names": self.names,
            "questionable": [bool(p.questionable) for p in self.parameters],
            "active": self.active.tolist(),
            "log_alpha": dict(
                zip([self.names[i] for i in self.ard_index], self.log_alpha.tolist())
            ),
            "relevance": dict(
                zip([self.names[i] for i in self.ard_index], self.relevance.tolist())
            ),
            "evidence_gain": dict(
                zip([self.names[i] for i in self.ard_index], self.evidence_gain.tolist())
            ),
            "relevance_tol": self.relevance_tol,
            "log_evidence": self.log_evidence,
            "posterior_mean": mean.tolist(),
            "posterior_std": std.tolist(),
            "n_optima": len(self.optima),
        }


@dataclass
class Calibration:
    """Posterior of one sparse model, sampled directly (no mixture approximation).

    Attributes
    ----------
    names : list of str
    active : ndarray of bool
        Parameters that were calibrated; the others are fixed at zero.
    samples : ndarray, shape (N, d)
        Posterior draws of the full parameter vector (zeros for pruned entries).
    log_likelihood : ndarray, shape (N,)
    log_evidence, log_evidence_error : float
        Evidence of this sparse model under uniform priors on the admissible range
        of its active questionable parameters, and a rough one-sigma error.
        Differences of ``log_evidence`` between structures are log Bayes factors.
    sampling : SMCResult
    """

    names: list[str]
    active: np.ndarray
    samples: np.ndarray
    log_likelihood: np.ndarray
    log_evidence: float
    log_evidence_error: float
    sampling: SMCResult

    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def std(self) -> np.ndarray:
        return self.samples.std(axis=0)

    def quantile(self, q: float | Sequence[float]) -> np.ndarray:
        return np.quantile(self.samples, q, axis=0)

    def best(self) -> np.ndarray:
        """Draw with the highest likelihood (the MAP under the uniform priors)."""
        return self.samples[int(np.argmax(self.log_likelihood))]

    def summary(self) -> str:
        mean, std = self.mean(), self.std()
        lo, hi = self.quantile([0.025, 0.975])
        width = max(max(len(n) for n in self.names), 9)
        lines = [
            f"Calibration of the sparse model ({int(self.active.sum())} active parameters): "
            f"log-evidence = {self.log_evidence:.3f} +- {self.log_evidence_error:.3f}",
            f"  {'parameter':<{width}}  {'mean':>11}  {'std':>10}  {'2.5%':>11}  {'97.5%':>11}",
        ]
        for i, name in enumerate(self.names):
            if self.active[i]:
                lines.append(
                    f"  {name:<{width}}  {mean[i]:>11.4g}  {std[i]:>10.3g}"
                    f"  {lo[i]:>11.4g}  {hi[i]:>11.4g}"
                )
        return "\n".join(lines)


def model_average(
    calibrations: Sequence[Calibration],
    n: int,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Posterior draws averaged over several sparse structures.

    Each structure contributes in proportion to its evidence (equal prior
    probabilities), which is Bayesian model averaging.  Predictions made with these
    draws carry the uncertainty on *which* terms belong in the model, not only on
    the coefficients of one structure; with a collinear dictionary the former is
    often the larger part.

    Parameters
    ----------
    calibrations : sequence of Calibration
        Typically the output of :meth:`NSBL.compare_structures`.
    n : int
        Number of draws.
    seed : int or numpy.random.Generator, optional

    Returns
    -------
    ndarray, shape (n, d)
    """
    rng = np.random.default_rng(seed)
    log_evidence = np.array([c.log_evidence for c in calibrations])
    weights = np.exp(log_evidence - log_evidence.max())
    weights /= weights.sum()
    counts = rng.multinomial(n, weights)
    draws = [
        c.samples[rng.integers(0, c.samples.shape[0], size=k)]
        for c, k in zip(calibrations, counts)
        if k > 0
    ]
    return rng.permutation(np.vstack(draws))


class NSBL:
    r"""Nonlinear sparse Bayesian learning for a black-box likelihood.

    Parameters
    ----------
    log_likelihood : callable
        :math:`\log p(\mathcal{D}\mid\phi)`.  With ``vectorized=True`` it receives an
        ``(n, d)`` array and returns ``n`` values; this is what makes the sampling
        stage fast, so vectorise it whenever the forward model allows.
    parameters : sequence of Parameter
        Questionable parameters (ARD prior) and a priori relevant ones (known prior).
    hyperprior : GammaHyperprior, optional
        Prior on the ARD precisions.  Flat in :math:`\log\alpha` by default, the
        Jeffreys limit that the paper approximates with small Gamma parameters;
        pass ``GammaHyperprior.paper()`` for its exact values.
    truncate_prior : bool
        ``False`` -- plain Gaussian ARD priors, the formulation of the paper; the
        bounds of the questionable parameters only delimit where the likelihood is
        evaluated.  ``True`` -- the ARD priors are truncated to the bounds and
        renormalised, which is the consistent choice when the bounds are genuine
        constraints (e.g. the design-of-experiments box of a surrogate): the model
        then tends to the uniform-prior model when a precision vanishes.
    standardize : bool
        If ``True`` (default) the hyperprior and the starting points act on the
        dimensionless precisions :math:`\alpha_i\,\mathrm{Var}(\phi_i)`, with the
        variance taken from the Gaussian mixture, which makes the result
        independent of the units of the parameters.  It matters as soon as the
        hyperprior has a positive rate: such a rate caps the precision a pruned
        parameter can reach, and without the scaling a coefficient that the data
        pin within :math:`10^{-2}` of zero is reported as relevant simply because
        its scale is small.  ``False`` reproduces the paper literally.
    vectorized : bool
        Set to ``False`` if ``log_likelihood`` takes one parameter vector at a time.
    """

    def __init__(
        self,
        log_likelihood: Callable[[np.ndarray], np.ndarray],
        parameters: Sequence[Parameter],
        *,
        hyperprior: GammaHyperprior | None = None,
        truncate_prior: bool = False,
        standardize: bool = True,
        vectorized: bool = True,
    ) -> None:
        self.parameters = list(parameters)
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("parameter names must be unique")
        self.dim = len(self.parameters)
        self.names = [p.name for p in self.parameters]
        self.lower = np.array([p.lower for p in self.parameters], dtype=float)
        self.upper = np.array([p.upper for p in self.parameters], dtype=float)
        self.ard_index = np.array(
            [i for i, p in enumerate(self.parameters) if p.questionable], dtype=int
        )
        if self.ard_index.size == 0:
            raise ValueError("no questionable parameter: there is nothing to sparsify")
        self.fold_index = np.array(
            [i for i in self.ard_index if self.lower[i] == 0.0 or self.upper[i] == 0.0], dtype=int
        )
        self.hyperprior = GammaHyperprior.none() if hyperprior is None else hyperprior
        self.truncate_prior = bool(truncate_prior)
        self.standardize = bool(standardize)

        if vectorized:
            self.log_likelihood = log_likelihood
        else:
            self.log_likelihood = lambda x: np.array([log_likelihood(row) for row in x])

        self.sampling_: SMCResult | None = None
        self.samples_: np.ndarray | None = None
        self.log_mass_: float = 0.0
        self.mixture_: GaussianMixture | None = None
        self.result_: NSBLResult | None = None

    # ------------------------------------------------------------------ #
    #  stage 1: sampling
    # ------------------------------------------------------------------ #
    def _sampling_prior(self, index: np.ndarray | None = None) -> JointPrior:
        index = np.arange(self.dim) if index is None else index
        priors = []
        for i in index:
            p = self.parameters[i]
            if p.questionable:
                if not (np.isfinite(p.lower) and np.isfinite(p.upper)):
                    raise ValueError(
                        f"questionable parameter '{p.name}' needs finite bounds for sampling"
                    )
                priors.append(Uniform(p.lower, p.upper))
            else:
                priors.append(p.prior)
        return JointPrior(priors)

    def sample(
        self,
        n_particles: int = 2000,
        *,
        seed: int | np.random.Generator | None = None,
        sampler: SMCSampler | None = None,
        **sampler_options,
    ) -> SMCResult:
        r"""Stage 1 -- draw from :math:`p(\mathcal{D}\mid\phi)\,p(\phi_{-\alpha})`.

        The questionable parameters are given a flat density on their admissible
        range, so the draws follow the likelihood times the known prior, restricted
        to that range.  Extra keyword arguments go to :class:`SMCSampler`.
        """
        if sampler is None:
            sampler = SMCSampler(n_particles, seed=seed, **sampler_options)
        result = sampler.run(self.log_likelihood, self._sampling_prior())
        volume = np.sum(np.log(self.upper[self.ard_index] - self.lower[self.ard_index]))
        self.sampling_ = result
        self.samples_ = result.samples
        self.log_mass_ = float(result.log_evidence + volume)
        self.mixture_ = None
        return result

    def set_samples(self, samples: np.ndarray, log_mass: float = 0.0) -> None:
        r"""Use draws obtained elsewhere (any MCMC sampler) instead of :meth:`sample`.

        Parameters
        ----------
        samples : array_like, shape (N, d)
            Stationary draws from :math:`p(\mathcal{D}\mid\phi)\,p(\phi_{-\alpha})`.
        log_mass : float
            :math:`\log\int p(\mathcal{D}\mid\phi)\,p(\phi_{-\alpha})\,d\phi` if
            known.  It only shifts the reported log-evidence by a constant.
        """
        samples = np.atleast_2d(np.asarray(samples, dtype=float))
        if samples.shape[1] != self.dim:
            raise ValueError(f"samples must have {self.dim} columns")
        self.samples_, self.log_mass_, self.mixture_ = samples, float(log_mass), None

    # ------------------------------------------------------------------ #
    #  stage 2: Gaussian mixture
    # ------------------------------------------------------------------ #
    def fit_mixture(self, method: str = "auto", **options) -> GaussianMixture:
        """Stage 2 -- Gaussian-mixture approximation of the sampled density.

        Parameters
        ----------
        method : {"auto", "kde", "em", "gaussian"}
            ``"kde"`` -- one kernel per sample, the construction of the paper.  It
            follows any shape and is exact at a zero bound, but beyond eight or so
            parameters the evidence of a sparse model rests on too few kernels and
            becomes noisy.
            ``"em"`` -- a few full-covariance kernels fitted by
            expectation-maximisation, which give a smooth evidence in high
            dimension.
            ``"gaussian"`` -- a single moment-matched kernel.
            ``"auto"`` (default) -- ``"kde"`` up to eight parameters, ``"em"``
            beyond.
        **options
            Forwarded to the constructor (see :class:`GaussianMixture`).  For
            ``"em"`` and ``"gaussian"``, ``symmetric_dims`` defaults to the
            sign-constrained parameters whose samples lie within two standard
            deviations of zero; they are mirrored about zero before the fit.
        """
        if self.samples_ is None:
            raise RuntimeError("call sample() or set_samples() first")
        x = self.samples_
        if method == "auto":
            method = "kde" if self.dim <= 8 else "em"
        if method in ("em", "gaussian") and "symmetric_dims" not in options:
            piled = x[:, self.fold_index]
            near_zero = np.abs(piled.mean(axis=0)) < 2.0 * piled.std(axis=0)
            options["symmetric_dims"] = self.fold_index[near_zero]
        if method == "kde":
            mixture = GaussianMixture.from_kde(x, log_mass=self.log_mass_, **options)
        elif method == "em":
            mixture = GaussianMixture.from_em(x, log_mass=self.log_mass_, **options)
        elif method == "gaussian":
            mixture = GaussianMixture.from_moments(x, log_mass=self.log_mass_, **options)
        else:
            raise ValueError(f"unknown mixture method '{method}'")
        self.mixture_ = mixture
        return mixture

    def set_mixture(self, mixture: GaussianMixture) -> None:
        """Use a mixture built elsewhere, e.g. the exact one of a linear-Gaussian model."""
        if mixture.dim != self.dim:
            raise ValueError(f"the mixture must have dimension {self.dim}")
        self.mixture_ = mixture

    # ------------------------------------------------------------------ #
    #  stage 3: evidence maximisation
    # ------------------------------------------------------------------ #
    def evidence(self) -> ARDEvidence:
        """The semi-analytical objective for the current mixture."""
        if self.mixture_ is None:
            raise RuntimeError("call fit_mixture() or set_mixture() first")
        hyperprior, shift = self.hyperprior, None
        if self.standardize:
            variance = np.diag(self.mixture_.covariance())[self.ard_index]
            # Gamma(r, s) on alpha * variance  <=>  Gamma(r, s * variance) on alpha
            hyperprior = GammaHyperprior(hyperprior.shape, hyperprior.rate * variance)
            shift = -np.log(variance)
        return ARDEvidence(
            self.mixture_,
            self.ard_index,
            theta_shift=shift,
            hyperprior=hyperprior,
            lower=self.lower[self.ard_index],
            upper=self.upper[self.ard_index],
            truncate=self.truncate_prior,
        )

    def optimize(
        self,
        *,
        n_starts: int = 16,
        starts: np.ndarray | None = None,
        relevance_tol: float = 0.5,
        seed: int | None = None,
        **options,
    ) -> NSBLResult:
        r"""Stage 3 -- multistart Newton maximisation of the evidence.

        Parameters
        ----------
        n_starts : int
            Number of starting values of :math:`\log\alpha`.
        starts : array_like, optional
            Explicit starting values, shape ``(n_runs, n_questionable)``.
        relevance_tol : float
            Threshold on the relevance indicator above which a parameter is kept.
        seed : int, optional
        **options
            Forwarded to :func:`sblcfd.optimize.maximize_evidence`.
        """
        objective = self.evidence()
        optima = maximize_evidence(
            objective, starts=starts, n_starts=n_starts, seed=seed, **options
        )
        if self.fold_index.size and not self.truncate_prior:
            # half-normal priors on the sign-constrained parameters: a factor 2 each,
            # which the truncated normalisation already contains
            offset = self.fold_index.size * np.log(2.0)
            for optimum in optima:
                optimum.log_evidence += offset
                optimum.objective += offset
        best = optima[0]
        self.result_ = NSBLResult(
            parameters=self.parameters,
            ard_index=self.ard_index,
            log_alpha=best.theta,
            relevance=best.relevance,
            relevance_tol=float(relevance_tol),
            evidence_gain=objective.evidence_gain(best.theta),
            log_evidence=best.log_evidence,
            objective=best.objective,
            posterior=objective.posterior(best.theta),
            optima=optima,
        )
        return self.result_

    def fit(
        self,
        n_particles: int = 2000,
        *,
        mixture: str = "auto",
        n_starts: int = 16,
        relevance_tol: float = 0.5,
        seed: int | None = None,
    ) -> NSBLResult:
        """Run the three stages with their default settings."""
        self.sample(n_particles, seed=seed)
        self.fit_mixture(mixture)
        return self.optimize(n_starts=n_starts, relevance_tol=relevance_tol, seed=seed)

    # ------------------------------------------------------------------ #
    #  stage 4 (optional): calibration of a sparse model
    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        active: Sequence[bool] | None = None,
        n_particles: int = 2000,
        *,
        seed: int | np.random.Generator | None = None,
        **sampler_options,
    ) -> Calibration:
        """Sample the posterior of a sparse model, pruned parameters fixed at zero.

        The active questionable parameters receive uniform priors on their
        admissible range, so the result is an ordinary Bayesian calibration of the
        selected structure, and its evidence can be compared across structures.

        Parameters
        ----------
        active : sequence of bool, optional
            Mask over all parameters.  Defaults to the model selected by the last
            call to :meth:`optimize`.  Known-prior parameters are always active.
        n_particles : int
        seed : int, optional
        **sampler_options
            Forwarded to :class:`SMCSampler`.
        """
        if active is None:
            if self.result_ is None:
                raise RuntimeError("call optimize() first or pass an explicit mask")
            active = self.result_.active
        active = np.asarray(active, dtype=bool).copy()
        if active.shape != (self.dim,):
            raise ValueError(f"the mask must have {self.dim} entries")
        active[[i for i, p in enumerate(self.parameters) if not p.questionable]] = True
        index = np.flatnonzero(active)

        def restricted(points: np.ndarray) -> np.ndarray:
            full = np.zeros((points.shape[0], self.dim))
            full[:, index] = points
            return self.log_likelihood(full)

        sampler = SMCSampler(n_particles, seed=seed, **sampler_options)
        result = sampler.run(restricted, self._sampling_prior(index))
        samples = np.zeros((result.samples.shape[0], self.dim))
        samples[:, index] = result.samples
        return Calibration(
            names=self.names,
            active=active,
            samples=samples,
            log_likelihood=result.log_likelihood,
            log_evidence=result.log_evidence,
            log_evidence_error=result.log_evidence_error,
            sampling=result,
        )

    def compare_structures(
        self,
        structures: Sequence[Sequence[bool]] | None = None,
        *,
        n_candidates: int = 5,
        n_particles: int = 2000,
        seed: int | None = None,
        **sampler_options,
    ) -> list[Calibration]:
        """Rank candidate sparse structures by their sampled evidence.

        The evidence maximisation searches the :math:`2^n` structures cheaply but
        through a Gaussian-mixture approximation.  This step calibrates a handful
        of candidates directly and sorts them by evidence, so that the final choice
        does not depend on that approximation.  Differences of ``log_evidence``
        between the returned calibrations are log Bayes factors.

        Parameters
        ----------
        structures : sequence of boolean masks, optional
            Candidates; defaults to the best ``n_candidates`` structures found by
            the last call to :meth:`optimize`.
        n_candidates : int
        n_particles : int
        seed : int, optional
        **sampler_options
            Forwarded to :class:`SMCSampler`.

        Returns
        -------
        list of Calibration
            Best first.
        """
        if structures is None:
            if self.result_ is None:
                raise RuntimeError("call optimize() first or pass explicit structures")
            structures = self.result_.structures(n_candidates)
        seeds = np.random.SeedSequence(seed).spawn(len(structures))
        calibrations = [
            self.calibrate(mask, n_particles, seed=np.random.default_rng(s), **sampler_options)
            for mask, s in zip(structures, seeds)
        ]
        return sorted(calibrations, key=lambda c: -c.log_evidence)
