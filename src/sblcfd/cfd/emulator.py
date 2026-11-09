r"""POD-Kriging emulator of a parametrised field.

.. math::

    Q(\Gamma, \mathbf{x}) \approx Q_0(\mathbf{x}) + \sum_{k=1}^{r} a_k(\Gamma)\,\Phi_k(\mathbf{x})

The modes :math:`\Phi_k` come from a proper orthogonal decomposition of the
snapshots and the coefficients :math:`a_k(\Gamma)` from Kriging.  The emulator can
be trained here (:meth:`PODEmulator.fit`) or taken over from a trained
``smt.applications.PODI`` object (:meth:`PODEmulator.from_smt`).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .kriging import KrigingPredictor, fit_kriging

__all__ = ["PODEmulator"]


class PODEmulator:
    """Reduced-order emulator ``field(x) = mean + modes @ coefficients(x)``.

    Parameters
    ----------
    mean : ndarray, shape (n_points,)
    modes : ndarray, shape (n_points, r)
    predictor : callable
        Maps an ``(n, D)`` array of inputs to the ``(n, r)`` modal coefficients;
        usually a :class:`KrigingPredictor`.
    energy : float, optional
        Fraction of the snapshot variance captured by the retained modes.
    """

    def __init__(
        self,
        mean: np.ndarray,
        modes: np.ndarray,
        predictor: Callable[[np.ndarray], np.ndarray],
        energy: float | None = None,
    ) -> None:
        self.mean = np.asarray(mean, dtype=float).ravel()
        self.modes = np.asarray(modes, dtype=float)
        if self.modes.ndim != 2 or self.modes.shape[0] != self.mean.size:
            raise ValueError("modes must have shape (n_points, r) matching the mean field")
        self.predictor = predictor
        self.energy = energy

    @property
    def n_points(self) -> int:
        return self.modes.shape[0]

    @property
    def n_modes(self) -> int:
        return self.modes.shape[1]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Input box covered by the training design (requires a Kriging predictor)."""
        return self.predictor.bounds()

    def coefficients(self, x: np.ndarray) -> np.ndarray:
        """Modal coefficients at the rows of ``x``; shape ``(n, r)``."""
        return np.asarray(self.predictor(np.atleast_2d(np.asarray(x, dtype=float))), dtype=float)

    def predict(self, x: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """Field values at the rows of ``x``; shape ``(n, n_points)`` or ``(n, len(rows))``."""
        coefficients = self.coefficients(x)
        if rows is None:
            return self.mean + coefficients @ self.modes.T
        rows = np.asarray(rows)
        return self.mean[rows] + coefficients @ self.modes[rows].T

    def validate(
        self,
        reference: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        rows: np.ndarray | None = None,
    ) -> float:
        """Largest relative deviation from a reference predictor on the rows of ``x``.

        ``reference`` returns the fields for a batch of inputs, as
        ``(n, n_points)`` or ``(n_points, n)``; use it to confirm that an emulator
        rebuilt from SMT reproduces ``podi.predict_values`` before trusting it.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        expected = _as_rows(np.asarray(reference(x), dtype=float), x.shape[0])
        if rows is not None and expected.shape[1] != np.asarray(rows).size:
            expected = expected[:, np.asarray(rows)]
        got = self.predict(x, rows)
        scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
        return float(np.max(np.abs(got - expected)) / scale)

    # ------------------------------------------------------------------ #
    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        snapshots: np.ndarray,
        *,
        energy: float = 0.999,
        n_modes: int | None = None,
        **kriging_options,
    ) -> PODEmulator:
        """Build the emulator from a design of experiments.

        Parameters
        ----------
        x : array_like, shape (n_samples, D)
            Inputs of the design of experiments.
        snapshots : array_like, shape (n_samples, n_points)
            One field per input.
        energy : float
            Modes are retained until this fraction of the snapshot variance is
            captured.
        n_modes : int, optional
            Fixed number of modes, overriding ``energy``.
        **kriging_options
            Forwarded to :func:`sblcfd.cfd.fit_kriging`.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        snapshots = np.asarray(snapshots, dtype=float)
        if snapshots.shape[0] != x.shape[0]:
            raise ValueError("one snapshot per row of x is required")
        mean = snapshots.mean(axis=0)
        centred = snapshots - mean
        # method of snapshots: the eigenproblem has the size of the design, not of the mesh
        values, vectors = np.linalg.eigh(centred @ centred.T)
        order = np.argsort(values)[::-1]
        values, vectors = np.maximum(values[order], 0.0), vectors[:, order]
        fraction = np.cumsum(values) / values.sum()
        if n_modes is None:
            n_modes = int(np.searchsorted(fraction, energy) + 1)
        n_modes = int(min(n_modes, np.count_nonzero(values > 1e-14 * values[0])))
        singular = np.sqrt(values[:n_modes])
        modes = centred.T @ vectors[:, :n_modes] / singular
        coefficients = vectors[:, :n_modes] * singular
        predictor = fit_kriging(x, coefficients, **kriging_options)
        return cls(mean, modes, predictor, energy=float(fraction[n_modes - 1]))

    @classmethod
    def from_smt(cls, podi) -> PODEmulator:
        """Take over a trained ``smt.applications.PODI`` model.

        Reads ``podi.basis``, ``podi.mean`` and the Kriging models in
        ``podi.interp_coeff``.  The result evaluates a whole population of inputs
        with a few matrix products, where ``podi.predict_values`` builds an
        ``(n x nt, D)`` array of differences per mode.
        """
        modes = np.asarray(podi.basis, dtype=float)
        mean = np.asarray(podi.mean, dtype=float).ravel()
        predictor = KrigingPredictor.from_smt(podi.interp_coeff)
        if modes.shape[1] != predictor.n_outputs:
            raise ValueError("the number of POD modes differs from the number of Kriging models")
        return cls(mean, modes, predictor)


def _as_rows(fields: np.ndarray, n_inputs: int) -> np.ndarray:
    """Return predictions as ``(n_inputs, n_points)`` whatever the layout of the surrogate."""
    if fields.ndim == 1:
        return fields.reshape(n_inputs, -1)
    if fields.shape[0] == n_inputs and fields.shape[1] != n_inputs:
        return fields
    if fields.shape[1] == n_inputs and fields.shape[0] != n_inputs:
        return fields.T
    if fields.shape[0] == n_inputs:  # square: ambiguous, assume one row per input
        return fields
    raise ValueError(f"cannot interpret predictions of shape {fields.shape} for {n_inputs} inputs")
