r"""Batched Kriging predictor and a compact maximum-likelihood trainer.

The sampling stage of NSBL evaluates the likelihood hundreds of thousands of
times, always for a whole population of parameter vectors at once.  What it needs
from a Gaussian-process emulator is therefore one thing: predicting ``K`` outputs
at ``n`` points with a few BLAS calls.  :class:`KrigingPredictor` does that for the
squared-exponential model used by SMT,

.. math::

    y_k(x) = \bar y_k + s_k\Big(f(\tilde x)^T\beta_k
             + \sum_{t} \gamma_{tk}\,
               \exp\big(-\textstyle\sum_l \eta_{lk}(\tilde x_l-\tilde x_{t,l})^2\big)\Big),
    \qquad \tilde x = (x - x_0)/x_s ,

and can be filled either from trained SMT models (:meth:`KrigingPredictor.from_smt`)
or by :func:`fit_kriging`, a dependency-free trainer for when SMT is not at hand.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import LinAlgError, cho_solve, cholesky, qr, solve_triangular
from scipy.optimize import minimize

__all__ = ["KrigingPredictor", "fit_kriging"]

_TRENDS = ("constant", "linear", "quadratic")


def _trend_basis(xn: np.ndarray, trend: str) -> np.ndarray:
    """Regression functions of SMT: constant, linear, or full quadratic."""
    n, dim = xn.shape
    if trend == "constant":
        return np.ones((n, 1))
    if trend == "linear":
        return np.hstack([np.ones((n, 1)), xn])
    if trend == "quadratic":
        cross = [xn[:, k] * xn[:, l] for k in range(dim) for l in range(k, dim)]
        return np.hstack([np.ones((n, 1)), xn, np.column_stack(cross)])
    raise ValueError(f"unknown trend '{trend}'; choose among {_TRENDS}")


class KrigingPredictor:
    """Squared-exponential Kriging predictor for ``K`` outputs sharing one design.

    Parameters
    ----------
    x_train : ndarray, shape (nt, D)
        Training inputs, already normalised.
    x_offset, x_scale : ndarray, shape (D,)
        Normalisation of the inputs.
    eta : ndarray, shape (D, K)
        Correlation weights of every output in the normalised input space.
    gamma : ndarray, shape (nt, K)
        Kriging weights :math:`R^{-1}(y - F\\beta)`.
    beta : ndarray, shape (n_f, K)
        Trend coefficients.
    y_mean, y_std : ndarray, shape (K,)
        Normalisation of the outputs.
    trend : {"constant", "linear", "quadratic"}
    """

    def __init__(
        self,
        x_train: np.ndarray,
        x_offset: np.ndarray,
        x_scale: np.ndarray,
        eta: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        trend: str = "constant",
    ) -> None:
        if trend not in _TRENDS:
            raise ValueError(f"unknown trend '{trend}'; choose among {_TRENDS}")
        self.x_train = np.asarray(x_train, dtype=float)
        self.x_offset = np.asarray(x_offset, dtype=float).ravel()
        self.x_scale = np.asarray(x_scale, dtype=float).ravel()
        self.eta = np.asarray(eta, dtype=float).reshape(self.x_train.shape[1], -1)
        self.gamma = np.asarray(gamma, dtype=float).reshape(self.x_train.shape[0], -1)
        self.beta = np.atleast_2d(np.asarray(beta, dtype=float))
        self.y_mean = np.asarray(y_mean, dtype=float).ravel()
        self.y_std = np.asarray(y_std, dtype=float).ravel()
        self.trend = trend
        if self.gamma.shape[1] != self.eta.shape[1]:
            raise ValueError("eta and gamma disagree on the number of outputs")
        # one correlation matrix serves every output when they share their weights
        self._shared = bool(np.allclose(self.eta, self.eta[:, :1]))
        root = np.sqrt(self.eta)
        self._root = root
        self._train = [
            self.x_train * root[:, k] for k in range(1 if self._shared else self.n_outputs)
        ]
        self._train_sq = [np.einsum("ij,ij->i", t, t) for t in self._train]

    @property
    def dim(self) -> int:
        return self.x_train.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.eta.shape[1]

    @property
    def n_train(self) -> int:
        return self.x_train.shape[0]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Box spanned by the training inputs: the region where the emulator can be trusted."""
        x = self.x_train * self.x_scale + self.x_offset
        return x.min(axis=0), x.max(axis=0)

    def _correlation(self, xn: np.ndarray, k: int) -> np.ndarray:
        scaled = xn * self._root[:, k]
        d2 = np.einsum("ij,ij->i", scaled, scaled)[:, None] + self._train_sq[k][None, :]
        d2 -= 2.0 * scaled @ self._train[k].T
        np.maximum(d2, 0.0, out=d2)
        return np.exp(-d2, out=d2)

    def predict(self, x: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
        """Predict the ``K`` outputs at the rows of ``x``; returns shape ``(n, K)``."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        xn = (x - self.x_offset) / self.x_scale
        out = _trend_basis(xn, self.trend) @ self.beta
        step = max(1, chunk // self.n_train)
        for start in range(0, x.shape[0], step):
            block = xn[start : start + step]
            if self._shared:
                out[start : start + step] += self._correlation(block, 0) @ self.gamma
            else:
                for k in range(self.n_outputs):
                    out[start : start + step, k] += self._correlation(block, k) @ self.gamma[:, k]
        return self.y_mean + self.y_std * out

    __call__ = predict

    # ------------------------------------------------------------------ #
    @classmethod
    def from_smt(cls, models: Sequence) -> KrigingPredictor:
        """Collect trained SMT ``KRG`` / ``KPLS`` / ``KPLSK`` models into one predictor.

        The models must share their training inputs and trend, as the modal
        coefficients of one ``smt.applications.PODI`` object do.  Only the
        squared-exponential correlation is supported.  For ``KPLS`` the weights are
        mapped back to the full space, :math:`\\eta_l = \\sum_k \\theta_k w_{lk}^2`.

        The attributes read here (``X_norma``, ``X_offset``, ``X_scale``,
        ``optimal_theta``, ``optimal_par``, ``y_mean``, ``y_std``, ``coeff_pls``)
        are SMT internals: always check the result against the original model with
        :meth:`sblcfd.cfd.PODEmulator.validate`.
        """
        models = list(models)
        if not models:
            raise ValueError("no SMT model given")
        columns = []
        for model in models:
            corr = model.options["corr"]
            if corr != "squar_exp":
                raise NotImplementedError(
                    f"only the 'squar_exp' correlation is supported, got '{corr}'"
                )
            x_norma = np.asarray(model.X_norma, dtype=float)
            dim = x_norma.shape[1]
            theta = np.asarray(model.optimal_theta, dtype=float).ravel()
            if theta.size == dim:
                eta = theta
            else:
                weights = np.asarray(model.coeff_pls, dtype=float)
                if weights.shape != (dim, theta.size):
                    raise ValueError(
                        f"PLS weights of shape {weights.shape} do not match {theta.size} thetas"
                    )
                eta = (weights**2) @ theta
            columns.append(
                {
                    "x": x_norma,
                    "offset": np.asarray(model.X_offset, dtype=float).ravel(),
                    "scale": np.asarray(model.X_scale, dtype=float).ravel(),
                    "eta": eta,
                    "beta": np.asarray(model.optimal_par["beta"], dtype=float).ravel(),
                    "gamma": np.asarray(model.optimal_par["gamma"], dtype=float).ravel(),
                    "y_mean": float(np.ravel(model.y_mean)[0]),
                    "y_std": float(np.ravel(model.y_std)[0]),
                    "trend": model.options["poly"],
                }
            )
        first = columns[0]
        for col in columns[1:]:
            same = (
                col["x"].shape == first["x"].shape
                and np.allclose(col["x"], first["x"])
                and np.allclose(col["offset"], first["offset"])
                and np.allclose(col["scale"], first["scale"])
                and col["trend"] == first["trend"]
            )
            if not same:
                raise ValueError("the SMT models do not share their training inputs and trend")
        return cls(
            first["x"],
            first["offset"],
            first["scale"],
            np.column_stack([c["eta"] for c in columns]),
            np.column_stack([c["gamma"] for c in columns]),
            np.column_stack([c["beta"] for c in columns]),
            np.array([c["y_mean"] for c in columns]),
            np.array([c["y_std"] for c in columns]),
            first["trend"],
        )


# --------------------------------------------------------------------------- #
#  training
# --------------------------------------------------------------------------- #
def _negative_log_likelihood(
    p: np.ndarray, d2: np.ndarray, basis: np.ndarray, y: np.ndarray, nugget: float
) -> tuple[float, np.ndarray, dict]:
    """Concentrated negative log-likelihood of ``log10(eta)`` and its gradient."""
    eta = 10.0**p
    n, k = y.shape
    corr = np.exp(-np.tensordot(d2, eta, axes=([2], [0])))
    corr[np.diag_indices(n)] += nugget
    chol = cholesky(corr, lower=True, check_finite=False)
    f_w = solve_triangular(chol, basis, lower=True, check_finite=False)
    y_w = solve_triangular(chol, y, lower=True, check_finite=False)
    q, r = qr(f_w, mode="economic")
    beta = solve_triangular(r, q.T @ y_w, check_finite=False)
    resid = y_w - f_w @ beta
    sigma2 = np.maximum(np.einsum("ij,ij->j", resid, resid) / n, np.finfo(float).tiny)
    value = 0.5 * (n * np.log(sigma2).sum() + 2.0 * k * np.log(np.diag(chol)).sum())

    weights = solve_triangular(chol.T, resid, lower=False, check_finite=False)  # R^-1 r
    inverse = cho_solve((chol, True), np.eye(n), check_finite=False)
    m = (weights / sigma2) @ weights.T - k * inverse
    grad_eta = 0.5 * np.tensordot(m * corr, d2, axes=([0, 1], [0, 1]))
    state = {"beta": beta, "gamma": weights, "sigma2": sigma2}
    return float(value), grad_eta * eta * np.log(10.0), state


def fit_kriging(
    x: np.ndarray,
    y: np.ndarray,
    *,
    trend: str = "constant",
    shared: bool = True,
    nugget: float = 1e-10,
    log10_eta_bounds: tuple[float, float] = (-6.0, 1.5),
    n_restarts: int = 1,
    seed: int | None = None,
) -> KrigingPredictor:
    r"""Train squared-exponential Kriging models by maximum likelihood.

    Parameters
    ----------
    x : array_like, shape (nt, D)
        Training inputs.
    y : array_like, shape (nt,) or (nt, K)
        Training outputs; ``K`` models are trained on the same inputs.
    trend : {"constant", "linear", "quadratic"}
    shared : bool
        If ``True`` all outputs share their correlation weights :math:`\eta`, which
        are found by a single likelihood maximisation.  Prediction then needs one
        correlation matrix for all outputs, so it is ``K`` times cheaper.  If
        ``False`` every output gets its own weights, as in SMT.
    nugget : float
        Diagonal regularisation of the correlation matrix.
    log10_eta_bounds : (float, float)
        Search range of :math:`\log_{10}\eta`.
    n_restarts : int
        Additional random starts of the optimiser.
    seed : int, optional
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)
    y = y[:, None] if y.ndim == 1 else y
    n, dim = x.shape
    if y.shape[0] != n:
        raise ValueError("x and y must have the same number of rows")

    x_offset, x_scale = x.mean(axis=0), x.std(axis=0, ddof=1)
    x_scale = np.where(x_scale > 0.0, x_scale, 1.0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0, ddof=1)
    y_std = np.where(y_std > 0.0, y_std, 1.0)
    xn, yn = (x - x_offset) / x_scale, (y - y_mean) / y_std
    basis = _trend_basis(xn, trend)
    d2 = (xn[:, None, :] - xn[None, :, :]) ** 2

    rng = np.random.default_rng(seed)
    lo, hi = log10_eta_bounds
    start = np.full(dim, np.clip(np.log10(0.5 / dim), lo, hi))

    def train(target: np.ndarray) -> tuple[np.ndarray, dict]:
        def objective(p: np.ndarray) -> tuple[float, np.ndarray]:
            jitter = nugget
            for _ in range(6):
                try:
                    value, grad, _ = _negative_log_likelihood(p, d2, basis, target, jitter)
                    return value, grad
                except LinAlgError:
                    jitter *= 100.0
            return 1e300, np.zeros_like(p)

        best = None
        for trial in range(1 + n_restarts):
            p0 = start if trial == 0 else rng.uniform(lo + 2.0, hi - 1.0, dim)
            res = minimize(objective, p0, jac=True, method="L-BFGS-B", bounds=[(lo, hi)] * dim)
            if best is None or res.fun < best.fun:
                best = res
        jitter = nugget
        while True:
            try:
                _, _, state = _negative_log_likelihood(best.x, d2, basis, target, jitter)
                break
            except LinAlgError:
                jitter *= 100.0
        return 10.0**best.x, state

    n_out = yn.shape[1]
    if shared:
        eta, state = train(yn)
        etas = np.repeat(eta[:, None], n_out, axis=1)
        gamma, beta = state["gamma"], state["beta"]
    else:
        etas, gamma = np.empty((dim, n_out)), np.empty((n, n_out))
        beta = np.empty((basis.shape[1], n_out))
        for k in range(n_out):
            etas[:, k], state = train(yn[:, k : k + 1])
            gamma[:, k], beta[:, k] = state["gamma"][:, 0], state["beta"][:, 0]
    return KrigingPredictor(xn, x_offset, x_scale, etas, gamma, beta, y_mean, y_std, trend)
