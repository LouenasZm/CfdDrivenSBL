"""Multistart Newton maximisation of the NSBL objective.

The objective is non-convex: distinct local optima correspond to distinct sparse
models (Section 3.2 of Sandhu et al., 2021).  Each start runs a trust-region
Newton iteration with the exact gradient and Hessian; the optima reached are then
merged and ranked, so the caller gets the list of competing sparse structures and
not only the best one.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import optimize as sopt
from scipy.stats import qmc

from .evidence import ARDEvidence

__all__ = ["Optimum", "default_starts", "maximize_evidence"]

_NEWTON_METHODS = {"trust-exact", "trust-ncg", "trust-krylov", "newton-cg", "dogleg"}
_BOUNDED_METHODS = {"L-BFGS-B", "TNC", "SLSQP"}


@dataclass
class Optimum:
    """One local optimum of the NSBL objective.

    Attributes
    ----------
    theta : ndarray
        :math:`\\log\\alpha` at the optimum.
    objective : float
        Log-evidence plus log-hyperprior.
    log_evidence : float
        Log-evidence alone.
    relevance : ndarray
        Relevance indicator of every questionable parameter.
    converged : bool
        Whether the Newton iteration met its tolerance.
    n_iterations : int
        Newton iterations of the run that found it.
    multiplicity : int
        Number of starts that converged to this optimum.
    """

    theta: np.ndarray
    objective: float
    log_evidence: float
    relevance: np.ndarray
    converged: bool
    n_iterations: int
    multiplicity: int = 1

    @property
    def alpha(self) -> np.ndarray:
        """ARD precisions at the optimum."""
        return np.exp(self.theta)


class _Memo:
    """Evaluate value, gradient and Hessian once per point; SciPy asks for them separately."""

    def __init__(self, objective: ARDEvidence, order: int) -> None:
        self.objective, self.order = objective, order
        self._key: bytes | None = None
        self._last = None

    def at(self, theta: np.ndarray):
        key = theta.tobytes()
        if key != self._key:
            self._last = self.objective.evaluate(theta, order=self.order)
            self._key = key
        return self._last


def default_starts(
    n: int, n_starts: int, theta_range: tuple[float, float], seed: int | None
) -> np.ndarray:
    """Starting points: the two extreme models, the centre, then a Sobol sequence.

    The first start has every precision small (all parameters relevant), the second
    every precision large (all parameters pruned); they are the starts used in the
    paper and they bracket the family of sparse models.
    """
    lo, hi = theta_range
    fixed = [np.full(n, lo + 0.1 * (hi - lo)), np.full(n, hi - 0.1 * (hi - lo)), np.zeros(n)]
    fixed = fixed[: max(n_starts, 1)]
    n_random = n_starts - len(fixed)
    if n_random <= 0:
        return np.array(fixed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Sobol balance for non powers of two
        unit = qmc.Sobol(d=n, scramble=True, seed=seed).random(n_random)
    return np.vstack([fixed, lo + (hi - lo) * unit])


def _local_search(
    objective: ARDEvidence,
    theta0: np.ndarray,
    free: np.ndarray,
    method: str,
    gtol: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, bool]:
    """Maximise over the coordinates flagged in ``free``, the others staying put."""
    index = np.flatnonzero(free)
    if index.size == 0:
        return theta0.copy(), 0, True
    newton = method in _NEWTON_METHODS
    memo = _Memo(objective, order=2 if newton else 1)

    def expand(z: np.ndarray) -> np.ndarray:
        theta = theta0.copy()
        theta[index] = z
        return theta

    kwargs: dict = {"jac": lambda z: -memo.at(expand(z)).gradient[index], "method": method}
    options: dict = {"maxiter": max_iterations, "gtol": gtol}
    if newton:
        kwargs["hess"] = lambda z: -memo.at(expand(z)).hessian[np.ix_(index, index)]
        if method != "newton-cg":
            # a step of ten in log(alpha) is already four orders of magnitude
            options.update(initial_trust_radius=1.0, max_trust_radius=10.0)
    elif method.upper() in _BOUNDED_METHODS:
        kwargs["bounds"] = list(zip(objective.theta_lower[index], objective.theta_upper[index]))
    res = sopt.minimize(
        lambda z: -memo.at(expand(z)).value, theta0[index], options=options, **kwargs
    )
    return expand(np.asarray(res.x, dtype=float)), int(getattr(res, "nit", 0)), bool(res.success)


def _prune_ties(objective: ARDEvidence, theta: np.ndarray, tol: float) -> np.ndarray:
    """Send to the pruned state every precision that the evidence does not clearly support.

    Two situations call for it.  Once an ARD prior is narrower than the kernels of
    the mixture the evidence is flat in its precision, and the point at which a
    local search stops on that plateau is decided by sampling noise.  And a
    precision fitted to the data always does at least as well as pruning, so a small
    gain is no evidence that the term is needed.  In both cases the tie is resolved
    in favour of the sparser model: precisions are moved to their upper limit, least
    relevant first, for as long as the objective stays within ``tol`` of the value
    reached by the local search.
    """
    theta = theta.copy()
    reference = objective(theta)
    for i in np.argsort(objective.relevance(theta)):
        if theta[i] >= objective.theta_upper[i]:
            continue
        trial = theta.copy()
        trial[i] = objective.theta_upper[i]
        if objective(trial) > reference - tol:
            theta = trial
    return theta


def maximize_evidence(
    objective: ARDEvidence,
    *,
    starts: np.ndarray | None = None,
    n_starts: int = 16,
    theta_range: tuple[float, float] = (-6.0, 6.0),
    method: str = "trust-exact",
    gtol: float = 1e-6,
    max_iterations: int = 500,
    evidence_tol: float = 1.0,
    relevance_tol: float = 0.05,
    seed: int | None = None,
) -> list[Optimum]:
    r"""Find the local maxima of :math:`\mathcal{L}(\log\alpha)`.

    Parameters
    ----------
    objective : ARDEvidence
    starts : array_like, shape (n_runs, n_ard), optional
        Explicit starting values of :math:`\log\alpha`; overrides ``n_starts``.
    n_starts : int
        Number of starts generated by :func:`default_starts`.
    theta_range : (float, float)
        Range covered by the generated starts, relative to the centre of the
        admissible interval of :math:`\log\alpha` (``objective.theta_shift``).
    method : str
        Any ``scipy.optimize.minimize`` method.  The trust-region Newton methods
        use the exact Hessian; ``"trust-exact"`` is the robust choice up to a few
        hundred questionable parameters, ``"trust-krylov"`` beyond.
    gtol : float
        Gradient tolerance of the local search.
    max_iterations : int
    evidence_tol : float
        Smallest gain in log-evidence (in nats) for which terms are kept.  After
        the local search, precisions are sent to their upper limit (the pruned
        state) as long as the objective drops by less than ``evidence_tol`` in
        total, and the remaining ones are re-optimised.  The default, one nat, is a
        Bayes factor of e: below it the data do not distinguish the two models.
        Zero keeps the raw local optimum, the type-II maximum-likelihood rule of
        the paper, which keeps any term whose estimate exceeds its own standard
        error.
    relevance_tol : float
        Two runs are merged when their relevance indicators differ by less than
        this everywhere and their objectives agree.  The precision of a pruned
        parameter is barely determined, so comparing relevance indicators is more
        meaningful than comparing precisions.
    seed : int, optional

    Returns
    -------
    list of Optimum
        Distinct optima, best first.
    """
    n = objective.n
    if starts is None:
        starts = default_starts(n, n_starts, theta_range, seed) + objective.theta_shift[None, :]
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    if starts.shape[1] != n:
        raise ValueError(f"each start needs {n} values, got {starts.shape[1]}")

    runs: list[Optimum] = []
    everything = np.ones(n, dtype=bool)
    for theta0 in starts:
        try:
            theta, n_iterations, success = _local_search(
                objective, theta0, everything, method, gtol, max_iterations
            )
            if evidence_tol > 0.0:
                snapped = _prune_ties(objective, theta, evidence_tol)
                pruned = snapped >= objective.theta_upper
                if not np.array_equal(snapped, theta):
                    # the kept precisions shift a little once the others are pruned
                    theta, extra, success = _local_search(
                        objective, snapped, ~pruned, method, gtol, max_iterations
                    )
                    n_iterations += extra
        except np.linalg.LinAlgError:  # pragma: no cover - defensive, the start is skipped
            continue
        final = objective.evaluate(theta, order=1)
        runs.append(
            Optimum(
                theta=theta,
                objective=float(final.value),
                log_evidence=float(final.log_evidence),
                relevance=objective.relevance(theta),
                converged=success,
                n_iterations=n_iterations,
            )
        )
    if not runs:
        raise RuntimeError("every start of the evidence maximisation failed")

    runs.sort(key=lambda run: -run.objective)
    distinct: list[Optimum] = []
    for run in runs:
        for kept in distinct:
            same_pattern = np.max(np.abs(run.relevance - kept.relevance)) < relevance_tol
            same_value = abs(run.objective - kept.objective) < 1e-3 * (1.0 + abs(kept.objective))
            if same_pattern and same_value:
                kept.multiplicity += 1
                break
        else:
            distinct.append(run)
    return distinct
