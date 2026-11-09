r"""Likelihood of field data predicted by reduced-order emulators.

Two ingredients make the likelihood cheap enough for population-based sampling.

*Reduced misfit.*  For a POD emulator :math:`Q = Q_0 + \Phi a(\Gamma)` the squared
misfit on the selected mesh points is a quadratic form in the ``r`` modal
coefficients,

.. math::

    \lVert y - Q_0 - \Phi a\rVert_W^2 = \lVert R\,a - t\rVert^2 + \mathrm{SSE}_{\min},

with :math:`W^{1/2}\Phi = QR` computed once.  A likelihood evaluation therefore
costs :math:`O(r^2)` instead of :math:`O(n_{\rm points}\, r)`: the field is never
reconstructed.

*Noise handled analytically.*  The error variance of every (case, quantity) pair is
integrated out, so quantities of different magnitudes are weighted automatically
and no nuisance parameter has to be sampled.

Spatial correlation
-------------------
Mesh points are not independent observations: the discrepancy between a RANS
model and a reference solution is correlated over a good part of the boundary
layer.  Treating ``n_points`` values as independent gives a posterior that is far
too narrow.  ``n_eff``, the effective number of independent observations of each
field, tempers the likelihood accordingly; :func:`effective_sample_size` estimates
it from a residual field.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.linalg import qr
from scipy.special import gammaln

from .emulator import PODEmulator, _as_rows

__all__ = ["FieldLikelihood", "FieldMisfit", "effective_sample_size"]

_LOG_2PI = float(np.log(2.0 * np.pi))


class FieldMisfit:
    r"""Squared misfit between one emulated field and its reference.

    Parameters
    ----------
    emulator : PODEmulator or callable
        A :class:`PODEmulator` (fast reduced evaluation) or any callable returning
        the field for a batch of inputs, as ``(n, n_points)`` or ``(n_points, n)``.
    reference : array_like
        Reference values, either on all the points of the emulator or already
        restricted to ``rows``.
    rows : array_like of int, optional
        Mesh points entering the misfit (e.g. the boundary-layer mask).  Defaults
        to all points.
    weights : array_like, optional
        Non-negative weights of the selected points, e.g. cell areas so that the
        misfit approximates an integral and does not depend on mesh clustering.
    name : str
        Label used in reports.
    n_eff : float, optional
        Effective number of independent observations.  Defaults to the number of
        selected points, which is the classical but usually over-confident choice.
    scale : float, optional
        Reference mean squared error used to normalise the misfit in reports,
        typically that of the baseline model.
    grid_index : array_like of int, optional
        Position of the selected points in the flattened structured grid, when it
        differs from ``rows`` (an emulator trained on a subset of the grid).  Only
        used by :meth:`residual_field`.
    """

    def __init__(
        self,
        emulator: PODEmulator | Callable[[np.ndarray], np.ndarray],
        reference: np.ndarray,
        *,
        rows: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        name: str = "field",
        n_eff: float | None = None,
        scale: float | None = None,
        grid_index: np.ndarray | None = None,
    ) -> None:
        reference = np.asarray(reference, dtype=float).ravel()
        self.rows = None if rows is None else np.asarray(rows, dtype=int)
        self.grid_index = None if grid_index is None else np.asarray(grid_index, dtype=int)
        if self.rows is not None and reference.size != self.rows.size:
            reference = reference[self.rows]
        if not np.all(np.isfinite(reference)):
            raise ValueError(
                f"'{name}': the reference contains non-finite values on the selected points"
            )
        self.reference = reference
        self.n_points = reference.size
        if weights is None:
            w = np.ones(self.n_points)
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.size != self.n_points:
                if self.rows is None:
                    raise ValueError(f"'{name}': one weight per reference value is required")
                w = w[self.rows]
            if np.any(w < 0.0):
                raise ValueError("weights must be non-negative")
        # weights are rescaled to sum to the number of points, so that the mean squared
        # error keeps its meaning and the tempering below is not affected by their units
        self.weights = w * (self.n_points / w.sum())
        self.name = name
        self.n_eff = float(self.n_points if n_eff is None else n_eff)
        if not 0.0 < self.n_eff <= self.n_points:
            raise ValueError(f"'{name}': n_eff must lie in (0, {self.n_points}]")
        self.scale = None if scale is None else float(scale)
        self.emulator = emulator

        self._reduced = isinstance(emulator, PODEmulator)
        if self._reduced:
            modes = emulator.modes if self.rows is None else emulator.modes[self.rows]
            mean = emulator.mean if self.rows is None else emulator.mean[self.rows]
            if modes.shape[0] != self.n_points:
                raise ValueError(f"'{name}': the reference does not match the selected points")
            root = np.sqrt(self.weights)
            target = root * (reference - mean)
            q, self._r = qr(root[:, None] * modes, mode="economic")
            self._t = q.T @ target
            self._sse_min = max(float(target @ target - self._t @ self._t), 0.0)

    def sse(self, x: np.ndarray, chunk: int = 4_000_000) -> np.ndarray:
        """Weighted sum of squared errors for every row of ``x``."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self._reduced:
            projected = self.emulator.coefficients(x) @ self._r.T - self._t
            return self._sse_min + np.einsum("ij,ij->i", projected, projected)
        out = np.empty(x.shape[0])
        step = max(1, chunk // self.n_points)
        for start in range(0, x.shape[0], step):
            block = x[start : start + step]
            fields = _as_rows(np.asarray(self.emulator(block), dtype=float), block.shape[0])
            if fields.shape[1] != self.n_points:
                fields = fields[:, self.rows]
            out[start : start + step] = ((fields - self.reference) ** 2) @ self.weights
        return out

    def mse(self, x: np.ndarray) -> np.ndarray:
        """Mean squared error for every row of ``x``."""
        return self.sse(x) / self.n_points

    def residual(self, x: np.ndarray) -> np.ndarray:
        """``model - reference`` on the selected points, for one parameter vector."""
        x = np.atleast_2d(np.asarray(x, dtype=float))[:1]
        if self._reduced:
            return self.emulator.predict(x, self.rows)[0] - self.reference
        fields = _as_rows(np.asarray(self.emulator(x), dtype=float), 1)
        if fields.shape[1] != self.n_points:
            fields = fields[:, self.rows]
        return fields[0] - self.reference

    def residual_field(self, x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Residual laid out on the structured grid, ``nan`` outside the selected points.

        This is the input :func:`effective_sample_size` expects.
        """
        full = np.full(int(np.prod(shape)), np.nan)
        if self.grid_index is not None:
            position = self.grid_index
        else:
            position = np.arange(self.n_points) if self.rows is None else self.rows
        full[position] = self.residual(x)
        return full.reshape(shape)

    @property
    def tempering(self) -> float:
        """Exponent ``n_eff / n_points`` applied to the likelihood of this field."""
        return self.n_eff / self.n_points


class FieldLikelihood:
    r"""Joint log-likelihood of several fields, vectorised over parameter vectors.

    Parameters
    ----------
    terms : sequence of FieldMisfit
        One per (flow case, quantity of interest).
    noise : {"marginal", "fixed"}
        ``"marginal"`` -- the error variance of each field has an inverse-Gamma
        prior and is integrated out,

        .. math:: \log L = -\sum_t \big(a_0 + \tfrac{n_t}{2}\big)
                  \log\big(b_0 + \tfrac{w_t}{2}\mathrm{SSE}_t\big) + \text{const},

        with :math:`n_t` the effective sample size and :math:`w_t` the tempering
        exponent.  With the default :math:`a_0 = b_0 = 0` (Jeffreys' prior) this is
        invariant to the scale of each quantity.
        ``"fixed"`` -- Gaussian errors of known standard deviation ``sigma``.
    sigma : float or array_like, optional
        Error standard deviation of each field, for ``noise="fixed"``.  Taking
        :math:`\sigma_t^2 = \kappa\,\mathrm{MSE}^{\rm baseline}_t` turns the
        posterior into a Gibbs posterior of the baseline-normalised loss used in
        deterministic CFD-driven training, with temperature set by :math:`\kappa`.
    prior_shape, prior_rate : float
        :math:`a_0` and :math:`b_0` of the inverse-Gamma prior.
    """

    def __init__(
        self,
        terms: Sequence[FieldMisfit],
        *,
        noise: str = "marginal",
        sigma: float | Sequence[float] | None = None,
        prior_shape: float = 0.0,
        prior_rate: float = 0.0,
    ) -> None:
        self.terms = list(terms)
        if not self.terms:
            raise ValueError("at least one field is required")
        if noise not in ("marginal", "fixed"):
            raise ValueError("noise must be 'marginal' or 'fixed'")
        self.noise = noise
        self.prior_shape, self.prior_rate = float(prior_shape), float(prior_rate)
        if noise == "fixed":
            if sigma is None:
                raise ValueError("noise='fixed' needs the error standard deviations")
            self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (len(self.terms),)).copy()
            if np.any(self.sigma <= 0.0):
                raise ValueError("sigma must be positive")
        else:
            self.sigma = None

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.terms]

    def sse(self, x: np.ndarray) -> np.ndarray:
        """Sum of squared errors of every field; shape ``(n, n_fields)``."""
        return np.column_stack([t.sse(x) for t in self.terms])

    def mse(self, x: np.ndarray) -> np.ndarray:
        """Mean squared error of every field; shape ``(n, n_fields)``."""
        return self.sse(x) / np.array([t.n_points for t in self.terms])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Log-likelihood for every row of ``x``."""
        sse = self.sse(x)
        # read at call time, so that n_eff can be refined between two runs
        n_eff = np.array([t.n_eff for t in self.terms])
        temper = np.array([t.tempering for t in self.terms])
        if self.noise == "fixed":
            quad = 0.5 * (temper * sse) / self.sigma**2
            const = n_eff * (np.log(self.sigma) + 0.5 * _LOG_2PI)
            return -(quad + const).sum(axis=1)
        shape = self.prior_shape + 0.5 * n_eff
        with np.errstate(divide="ignore"):
            value = -shape * np.log(self.prior_rate + 0.5 * temper * sse)
        const = gammaln(shape) - 0.5 * n_eff * _LOG_2PI
        if self.prior_shape > 0.0 and self.prior_rate > 0.0:
            const = const + self.prior_shape * np.log(self.prior_rate) - gammaln(self.prior_shape)
        return (value + const).sum(axis=1)

    # ------------------------------------------------------------------ #
    #  reporting helpers
    # ------------------------------------------------------------------ #
    def _scales(self) -> np.ndarray:
        if any(t.scale is None for t in self.terms):
            raise ValueError("every field needs a 'scale' (e.g. the baseline mean squared error)")
        return np.array([t.scale for t in self.terms])

    def normalised_loss(self, x: np.ndarray) -> np.ndarray:
        """Mean over the fields of ``MSE / scale``: the deterministic training loss."""
        return (self.mse(x) / self._scales()).mean(axis=1)

    def improvement(self, x: np.ndarray) -> np.ndarray:
        """``1 - MSE / scale`` of every field; shape ``(n, n_fields)``.

        With the baseline error as scale this is the improvement score: one for a
        perfect match, zero for no change, negative for a degradation.
        """
        return 1.0 - self.mse(x) / self._scales()

    def noise_level(self, x: np.ndarray) -> np.ndarray:
        """Root mean squared error of every field; shape ``(n, n_fields)``."""
        return np.sqrt(self.mse(x))


def effective_sample_size(residual: np.ndarray, floor: float = 0.05) -> float:
    r"""Effective number of independent values in a spatially correlated residual.

    For a correlated Gaussian field the sum of squares :math:`\sum_i r_i^2` has the
    variance of a chi-square variable with

    .. math:: n_{\rm eff} = \frac{n}{\prod_{\rm axes}\big(1 + 2\sum_{k\ge1}\rho_k^2\big)}

    degrees of freedom, :math:`\rho_k` being the autocorrelation at lag ``k`` along
    an axis (separable approximation).  The autocorrelations are taken about zero,
    not about the mean, so a uniform bias counts as one single piece of information.

    Parameters
    ----------
    residual : array_like, one- or two-dimensional
        Residual field on a structured grid, ``model - reference`` for a
        representative parameter vector (the baseline is a natural choice).  Points
        outside the region of interest are marked with ``nan``.
    floor : float
        The sum over lags stops at the first autocorrelation below this value.

    Returns
    -------
    float
        Between one and the number of valid points.
    """
    r = np.asarray(residual, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    if r.ndim != 2:
        raise ValueError("the residual must be a one- or two-dimensional array")
    valid = np.isfinite(r)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("the residual has no valid point")
    filled = np.where(valid, r, 0.0)
    energy = (filled**2).sum() / n_valid
    if energy == 0.0:
        return float(n_valid)

    inflation = 1.0
    for axis in (0, 1):
        data = np.moveaxis(filled, axis, 0)
        mask = np.moveaxis(valid, axis, 0)
        total = 0.0
        for lag in range(1, data.shape[0]):
            pairs = mask[:-lag] & mask[lag:]
            count = int(pairs.sum())
            if count < 2:
                break
            rho = float((data[:-lag] * data[lag:])[pairs].sum() / count / energy)
            if rho < floor:
                break
            total += min(rho, 1.0) ** 2
        inflation *= 1.0 + 2.0 * total
    return float(np.clip(n_valid / inflation, 1.0, n_valid))
