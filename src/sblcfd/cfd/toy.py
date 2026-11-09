r"""A miniature of CFD-driven transition modelling, cheap enough for tests and demos.

The model integrates an intermittency equation along a flat plate,

.. math::

    \frac{d\gamma}{dx} = c_P\,(F + R^P)\,(1-\gamma)\sqrt{\gamma}
                       - c_E\,(G + R^E)\,F_{\rm turb}\,\gamma^{3/2},

with baseline ramp functions :math:`F`, :math:`G` and the corrections
:math:`R^P`, :math:`R^E` of :func:`sblcfd.cfd.transition_library`, whose features
depend on the solution itself.  The intermittency then blends laminar and turbulent
velocity profiles into a two-dimensional boundary-layer field.

It is *not* a transition model.  It reproduces the structure of the real problem:
a nonlinear, coupled map from 21 coefficients to mean-flow fields, several flow
cases with and without pressure gradient, coefficients that cannot be identified
from every case, and a collinear dictionary.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .library import CorrectionLibrary, transition_library

__all__ = ["ToyTransition", "toy_cases"]


def _positive(z: np.ndarray, sharpness: float = 25.0) -> np.ndarray:
    """Smooth version of ``max(z, 0)``: activation functions cannot become negative."""
    return np.logaddexp(0.0, sharpness * z) / sharpness


class ToyTransition:
    """Flat-plate boundary layer with an augmented intermittency equation.

    Parameters
    ----------
    onset : float
        Critical value of the scaled vorticity Reynolds number at which the
        baseline production switches on; lower for higher free-stream turbulence.
    pressure_gradient : callable or float
        Acceleration parameter :math:`\\lambda(x)` on ``x in [0, 1]``, of order one;
        zero for a zero-pressure-gradient case.
    n_x, n_y : int
        Resolution of the output field.
    reynolds : float
        Reynolds number based on the plate length.
    library : CorrectionLibrary, optional
        Dictionary of corrections; defaults to :func:`transition_library`.
    """

    def __init__(
        self,
        onset: float = 0.7,
        pressure_gradient: Callable[[np.ndarray], np.ndarray] | float = 0.0,
        *,
        n_x: int = 64,
        n_y: int = 24,
        reynolds: float = 4.0e5,
        library: CorrectionLibrary | None = None,
    ) -> None:
        self.onset = float(onset)
        self.library = transition_library() if library is None else library
        self.n_x, self.n_y = int(n_x), int(n_y)
        self.x = np.linspace(0.04, 1.0, self.n_x)
        self._substeps = 6
        self._x_fine = np.linspace(0.0, 1.0, self._substeps * (self.n_x - 1) * 2 + 1)
        if callable(pressure_gradient):
            self._lam = pressure_gradient
        else:
            value = float(pressure_gradient)
            self._lam = lambda x: np.full_like(np.asarray(x, dtype=float), value)

        re_x = reynolds * self.x
        self.cf_laminar = 0.664 / np.sqrt(re_x)
        self.cf_turbulent = 0.0592 / re_x**0.2
        delta_lam = 5.0 * self.x / np.sqrt(re_x)
        delta_turb = 0.37 * self.x / re_x**0.2
        self.y = delta_turb[-1] * 1.3 * np.linspace(0.0, 1.0, self.n_y + 1)[1:] ** 1.5
        eta_l = self.y[None, :] / delta_lam[:, None]
        eta_t = self.y[None, :] / delta_turb[:, None]
        self._u_lam = np.tanh(2.2 * eta_l)
        self._u_turb = 0.55 * np.tanh(14.0 * eta_t) + 0.45 * np.tanh(2.0 * eta_t)
        self._du_lam = 2.2 / delta_lam[:, None] / np.cosh(2.2 * eta_l) ** 2
        self._du_turb = (
            0.55 * 14.0 / np.cosh(14.0 * eta_t) ** 2 + 0.45 * 2.0 / np.cosh(2.0 * eta_t) ** 2
        ) / delta_turb[:, None]
        scale = self._du_turb.max()
        self._du_lam, self._du_turb = self._du_lam / scale, self._du_turb / scale

    @property
    def n_parameters(self) -> int:
        return self.library.n_parameters

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(n_x, n_y)`` of the two-dimensional fields."""
        return self.n_x, self.n_y

    # ------------------------------------------------------------------ #
    def _rate(self, gamma: np.ndarray, x: float, coefficients: np.ndarray) -> np.ndarray:
        lam = float(self._lam(np.array(x)))
        vorticity = np.sqrt(x)  # scaled vorticity Reynolds number, grows like sqrt(Re_x)
        features = {
            "lam": lam,
            "eta1": gamma**2,  # eddy-viscosity ratio: zero in laminar flow
            "eta2": 1.0 / (1.0 + 4.0 * gamma),  # wall-distance ratio: one in laminar flow
            "eta3": vorticity,
        }
        corrections = self.library.evaluate(coefficients, features)
        ramp = 1.0 / (1.0 + np.exp(-(vorticity - self.onset - 0.08 * lam) / 0.05))
        damping = 0.25 * (1.0 - ramp)
        f_turb = np.exp(-((features["eta1"] / 0.6) ** 4))
        production = _positive(ramp + corrections["RP"]) * (1.0 - gamma) * np.sqrt(gamma)
        destruction = _positive(damping + corrections["RE"]) * f_turb * gamma**1.5
        return 22.0 * production - 9.0 * destruction

    def intermittency(self, coefficients: np.ndarray) -> np.ndarray:
        """Intermittency at the output stations; shape ``(n, n_x)``."""
        coefficients = np.atleast_2d(np.asarray(coefficients, dtype=float))
        if coefficients.shape[1] != self.n_parameters:
            raise ValueError(f"expected {self.n_parameters} coefficients per row")
        gamma = np.full(coefficients.shape[0], 0.02)
        out = np.empty((coefficients.shape[0], self.n_x))
        station = 0
        for i in range(self._x_fine.size - 1):
            x0, x1 = self._x_fine[i], self._x_fine[i + 1]
            while station < self.n_x and self.x[station] <= x0 + 1e-12:
                out[:, station] = gamma
                station += 1
            h = x1 - x0
            mid = np.clip(gamma + 0.5 * h * self._rate(gamma, x0, coefficients), 1e-4, 1.0)
            gamma = np.clip(gamma + h * self._rate(mid, x0 + 0.5 * h, coefficients), 1e-4, 1.0)
        out[:, station:] = gamma[:, None]
        return out

    def skin_friction(self, coefficients: np.ndarray) -> np.ndarray:
        """Skin-friction coefficient along the plate; shape ``(n, n_x)``."""
        gamma = self.intermittency(coefficients)
        return (1.0 - gamma) * self.cf_laminar + gamma * self.cf_turbulent

    def fields(self, coefficients: np.ndarray) -> dict[str, np.ndarray]:
        """Mean velocity ``u`` and its wall-normal derivative ``dudy``.

        Each entry has shape ``(n, n_x * n_y)``, the fields being flattened in
        C order as surrogates trained on CFD snapshots would store them.
        """
        gamma = self.intermittency(coefficients)[:, :, None]
        u = (1.0 - gamma) * self._u_lam + gamma * self._u_turb
        dudy = (1.0 - gamma) * self._du_lam + gamma * self._du_turb
        n = gamma.shape[0]
        return {"u": u.reshape(n, -1), "dudy": dudy.reshape(n, -1)}

    def boundary_layer_mask(self, coefficients: np.ndarray, threshold: float = 0.99) -> np.ndarray:
        """Flattened boolean mask of the points inside the boundary layer."""
        return self.fields(coefficients)["u"][0] < threshold


def toy_cases(n_x: int = 64, n_y: int = 24) -> dict[str, ToyTransition]:
    """Five flow cases in the spirit of the ERCOFTAC T3 series.

    ``zpg-low`` and ``zpg-high`` have no pressure gradient and differ by their
    free-stream turbulence.  ``pg-mixed`` goes from a favourable to an adverse
    pressure gradient, ``pg-adverse`` is adverse throughout and ``pg-favourable``
    favourable throughout; only these three carry information on the
    pressure-gradient terms of the dictionary.
    """
    return {
        "zpg-low": ToyTransition(onset=0.72, n_x=n_x, n_y=n_y),
        "zpg-high": ToyTransition(onset=0.52, n_x=n_x, n_y=n_y),
        "pg-mixed": ToyTransition(
            onset=0.62, pressure_gradient=lambda x: 0.6 * (1.0 - 2.0 * x), n_x=n_x, n_y=n_y
        ),
        "pg-adverse": ToyTransition(
            onset=0.62, pressure_gradient=lambda x: -0.1 - 0.5 * x, n_x=n_x, n_y=n_y
        ),
        "pg-favourable": ToyTransition(onset=0.55, pressure_gradient=0.4, n_x=n_x, n_y=n_y),
    }
