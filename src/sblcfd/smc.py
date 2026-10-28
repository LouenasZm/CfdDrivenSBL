"""Vectorised sequential Monte Carlo sampler (transitional MCMC).

NSBL needs stationary draws from ``likelihood x known prior`` exactly once.  The
sampler below follows the tempering scheme of TMCMC (Ching and Chen, 2007), the
algorithm used in the NSBL paper, with choices made for speed and accuracy:

* every likelihood call receives the *whole* population as one ``(N, d)`` array, so
  a surrogate-based likelihood is evaluated with a handful of BLAS calls per
  Markov step rather than ``N`` Python calls;
* two Metropolis kernels are combined.  Independence moves, drawn from a Gaussian
  fitted to the population, refresh a particle completely when accepted and are
  used for as long as they are accepted often enough, which is the case whenever
  the tempered target is close to Gaussian.  Random-walk moves, reflected at the
  bounds, take over otherwise and always close a stage;
* a stage ends when the population has decorrelated from its state at the start of
  the stage, so easy stages cost a few sweeps and hard ones as many as allowed;
* the population is split in two halves and each half is moved with proposals
  fitted on the *other* half.  Tuning the proposal on the particles it then moves
  biases the evidence upwards by ``O(d^2 / N)`` -- a factor of almost two for 21
  parameters and 1000 particles -- and this cross-fitting removes it at no cost.
  The two halves also give two independent evidence estimates, whose spread is
  reported as an error indicator.

It handles multimodal targets and returns the model evidence as a by-product.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from scipy.special import logsumexp

from .priors import JointPrior

__all__ = ["SMCResult", "SMCSampler"]


@dataclass
class SMCResult:
    """Output of :class:`SMCSampler`.

    Attributes
    ----------
    samples : ndarray, shape (N, d)
        Equally weighted draws from the posterior ``likelihood x prior``.
    log_likelihood, log_prior : ndarray, shape (N,)
        Values at the draws.
    log_evidence : float
        Estimate of :math:`\\log\\int L(\\phi)\\,p(\\phi)\\,d\\phi`.
    log_evidence_error : float
        Half the gap between the estimates of the two half-populations: a rough
        one-sigma error of ``log_evidence``.
    betas : list of float
        Tempering exponents visited, from 0 to 1.
    acceptance : list of float
        Mean Metropolis acceptance rate of each stage.
    correlation : list of float
        Largest correlation, over the coordinates, between the population at the
        start and at the end of each stage.  Values near one flag poor mixing.
    n_evaluations : int
        Total number of likelihood evaluations.
    wall_time : float
        Elapsed seconds.
    """

    samples: np.ndarray
    log_likelihood: np.ndarray
    log_prior: np.ndarray
    log_evidence: float
    log_evidence_error: float = 0.0
    betas: list[float] = field(default_factory=list)
    acceptance: list[float] = field(default_factory=list)
    correlation: list[float] = field(default_factory=list)
    n_evaluations: int = 0
    wall_time: float = 0.0

    @property
    def n_stages(self) -> int:
        return len(self.betas) - 1


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def _gaussian_fit(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and Cholesky factor of the covariance (diagonal if rank deficient)."""
    mean = w @ x
    dev = x - mean
    cov = (dev * w[:, None]).T @ dev
    dim = cov.shape[0]
    cov += 1e-12 * max(np.trace(cov) / dim, np.finfo(float).tiny) * np.eye(dim)
    try:
        return mean, cholesky(cov, lower=True, check_finite=False)
    except LinAlgError:
        return mean, np.diag(np.sqrt(np.diag(cov)))


def _max_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute column-wise correlation between two populations."""
    da, db = a - a.mean(axis=0), b - b.mean(axis=0)
    denom = np.sqrt((da * da).sum(axis=0) * (db * db).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0.0, (da * db).sum(axis=0) / denom, 1.0)
    return float(np.max(np.abs(corr)))


class SMCSampler:
    """Adaptive tempered sequential Monte Carlo with Metropolis moves.

    Parameters
    ----------
    n_particles : int
        Population size ``N``; also the number of posterior draws returned.
    ess_fraction : float
        Each stage raises the tempering exponent until the effective sample size
        of the importance weights drops to this fraction of ``N``.
    max_correlation : float
        A stage ends once every coordinate of the population is correlated by
        less than this with its value at the start of the stage.
    min_steps, max_steps : int
        Bounds on the number of Markov steps per stage.  ``min_steps`` random-walk
        steps are always taken; ``max_steps`` caps the cost of stages that cannot
        decorrelate, e.g. on multimodal targets.
    independent_moves : bool
        Allow independence moves from a Gaussian fitted to the population.
    independent_min_acceptance : float
        Independence moves are abandoned for the rest of a stage when their
        acceptance rate falls below this value.
    target_acceptance : float
        Acceptance rate the random-walk scale is steered towards.
    seed : int or numpy.random.Generator, optional
    verbose : bool
        Print one line per stage.
    """

    def __init__(
        self,
        n_particles: int = 2000,
        *,
        ess_fraction: float = 0.5,
        max_correlation: float = 0.25,
        min_steps: int = 2,
        max_steps: int = 40,
        independent_moves: bool = True,
        independent_min_acceptance: float = 0.08,
        target_acceptance: float = 0.234,
        seed: int | np.random.Generator | None = None,
        verbose: bool = False,
    ) -> None:
        if not 0.0 < ess_fraction < 1.0:
            raise ValueError("ess_fraction must lie strictly between 0 and 1")
        if n_particles < 4:
            raise ValueError("at least four particles are needed")
        self.n_particles = int(n_particles)
        self.ess_fraction = float(ess_fraction)
        self.max_correlation = float(max_correlation)
        self.min_steps, self.max_steps = int(min_steps), max(int(max_steps), int(min_steps))
        self.independent_moves = bool(independent_moves)
        self.independent_min_acceptance = float(independent_min_acceptance)
        self.target_acceptance = float(target_acceptance)
        self.rng = np.random.default_rng(seed)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------ #
    def _next_beta(self, ll: np.ndarray, beta: float) -> float:
        finite = np.isfinite(ll)
        target = self.ess_fraction * finite.sum()
        shifted = ll[finite] - ll[finite].max()

        def ess(delta: float) -> float:
            lw = delta * shifted
            lw -= logsumexp(lw)
            return float(np.exp(-logsumexp(2.0 * lw)))

        full = 1.0 - beta
        if ess(full) >= target:
            return 1.0
        lo, hi = 0.0, full
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if ess(mid) > target:
                lo = mid
            else:
                hi = mid
        return beta + (lo if lo > 0.0 else hi)

    def run(
        self,
        log_likelihood: Callable[[np.ndarray], np.ndarray],
        prior: JointPrior,
    ) -> SMCResult:
        """Sample ``likelihood x prior``.

        Parameters
        ----------
        log_likelihood : callable
            Maps an ``(n, d)`` array of parameter vectors to ``n`` log-likelihood
            values.  Non-finite values are treated as zero likelihood.
        prior : JointPrior
            Proper prior the population is initialised from.
        """
        start = time.perf_counter()
        rng, n, dim = self.rng, self.n_particles, prior.dim
        halves = (np.arange(n // 2), np.arange(n // 2, n))
        other = (1, 0)

        def loglik(points: np.ndarray) -> np.ndarray:
            values = np.asarray(log_likelihood(points), dtype=float).reshape(-1)
            if values.shape[0] != points.shape[0]:
                raise ValueError(
                    "log_likelihood must return one value per row of its input; "
                    f"got {values.shape[0]} for {points.shape[0]} rows"
                )
            return np.where(np.isfinite(values), values, -np.inf)

        x = prior.sample(n, rng)
        lp = prior.logpdf(x)
        ll = loglik(x)
        n_eval = n
        if not all(np.isfinite(ll[h]).any() for h in halves):
            raise RuntimeError(
                "the likelihood is zero for (almost) every draw of the prior; "
                "check the bounds or increase the number of particles"
            )

        beta = 0.0
        log_z = np.zeros(2)  # one evidence estimate per half-population
        betas, acceptance, correlation = [0.0], [], []
        scale = 2.38 / np.sqrt(dim)
        scale_cap = max(1.0, scale)
        means = [np.zeros(dim), np.zeros(dim)]
        factors = [np.eye(dim), np.eye(dim)]

        def log_reference(points: np.ndarray, h: int) -> np.ndarray:
            """Unnormalised log-density of the Gaussian used to move half ``h``."""
            z = solve_triangular(
                factors[other[h]], (points - means[other[h]]).T, lower=True, check_finite=False
            )
            return -0.5 * np.einsum("ij,ij->j", z, z)

        while beta < 1.0:
            new_beta = self._next_beta(ll, beta)
            log_w = (new_beta - beta) * ll
            beta = new_beta

            keep = np.empty(n, dtype=int)
            for h, members in enumerate(halves):
                log_norm = logsumexp(log_w[members])
                log_z[h] += log_norm - np.log(members.size)
                w = np.exp(log_w[members] - log_norm)
                means[h], factors[h] = _gaussian_fit(x[members], w)
                keep[members] = members[_systematic_resample(w, rng)]
            x, ll, lp = x[keep], ll[keep], lp[keep]
            x_start = x.copy()

            rates: list[float] = []
            independent_ok = self.independent_moves
            n_walk, corr = 0, 1.0
            for step in range(self.max_steps):
                # independence moves while they pay off, random-walk moves otherwise;
                # the last `min_steps` steps of a stage are always random-walk moves
                remaining = self.max_steps - step
                independent = (
                    independent_ok
                    and corr > self.max_correlation
                    and remaining > self.min_steps - n_walk
                )
                noise = rng.standard_normal((n, dim))
                steps = np.empty_like(x)
                # each half is moved with the Gaussian fitted on the other half
                for h, members in enumerate(halves):
                    steps[members] = noise[members] @ factors[other[h]].T
                hastings = np.zeros(n)
                if independent:
                    proposal = np.empty_like(x)
                    for h, members in enumerate(halves):
                        proposal[members] = means[other[h]] + steps[members]
                        hastings[members] = log_reference(x[members], h) - log_reference(
                            proposal[members], h
                        )
                else:
                    proposal, flipped = prior.reflect(x + scale * steps)
                    # a reflected step is the mirror image of the reverse step only if
                    # the reflected coordinates are uncorrelated with the others
                    for h, members in enumerate(halves):
                        rows = members[flipped[members].any(axis=1)]
                        if rows.size:
                            mirrored = np.where(flipped[rows], -steps[rows], steps[rows])
                            z = solve_triangular(
                                factors[other[h]], mirrored.T, lower=True, check_finite=False
                            )
                            hastings[rows] = 0.5 * (
                                np.einsum("ij,ij->i", noise[rows], noise[rows])
                                - np.einsum("ij,ij->j", z, z)
                            )
                lp_new = prior.logpdf(proposal)
                valid = np.isfinite(lp_new)
                ll_new = np.full(n, -np.inf)
                if valid.any():
                    ll_new[valid] = loglik(proposal[valid])
                    n_eval += int(valid.sum())
                with np.errstate(invalid="ignore"):
                    log_ratio = beta * (ll_new - ll) + lp_new - lp + hastings
                accept = np.log(rng.random(n)) < np.nan_to_num(log_ratio, nan=-np.inf)
                x[accept], ll[accept], lp[accept] = proposal[accept], ll_new[accept], lp_new[accept]
                rate = float(accept.mean())
                rates.append(rate)
                if independent:
                    independent_ok = rate >= self.independent_min_acceptance
                else:
                    n_walk += 1
                    scale *= float(
                        np.exp(np.clip(1.5 * (rate - self.target_acceptance), -0.7, 0.7))
                    )
                    # steps much longer than the population spread would wrap around
                    # a bounded support several times, where the correction above
                    # (which keeps a single mirror image) stops being exact
                    scale = min(scale, scale_cap)
                corr = _max_correlation(x_start, x)
                if n_walk >= self.min_steps and corr <= self.max_correlation:
                    break

            betas.append(beta)
            acceptance.append(float(np.mean(rates)))
            correlation.append(corr)
            if self.verbose:
                print(
                    f"  stage {len(betas) - 1:3d}  beta={beta:9.3e}  steps={len(rates):3d} "
                    f"({n_walk} random-walk)  acceptance={acceptance[-1]:.2f}  "
                    f"correlation={corr:.2f}  log-evidence={logsumexp(log_z) - np.log(2):.3f}"
                )

        return SMCResult(
            samples=x,
            log_likelihood=ll,
            log_prior=lp,
            log_evidence=float(logsumexp(log_z) - np.log(2.0)),
            log_evidence_error=float(0.5 * abs(log_z[0] - log_z[1])),
            betas=betas,
            acceptance=acceptance,
            correlation=correlation,
            n_evaluations=n_eval,
            wall_time=time.perf_counter() - start,
        )
