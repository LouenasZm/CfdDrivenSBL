"""Bridge to the data layout of the deterministic CFD-driven framework.

The elastic-net / bi-level framework (``AugTransitionRANS``) organises its inputs as

.. code-block:: python

    data = {
        "surrogates": {case: {variable: surrogate}},          # surrogate.predict(x)
        "les":        {case: {variable: field, "S_star": field}},
        "baseline":   {"loss": {case: {variable: mse}}},      # or {case: {variable: mse}}
    }

and evaluates a masked, baseline-normalised mean squared error.  The function below
turns the same dictionary into a :class:`~sblcfd.cfd.FieldLikelihood`, with the same
points entering the misfit, so both approaches can be run on identical inputs.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .emulator import PODEmulator, _as_rows
from .likelihood import FieldLikelihood, FieldMisfit

__all__ = ["likelihood_from_bilevel_data", "masked_rows"]


def masked_rows(
    reference: np.ndarray, s_star: np.ndarray, threshold: float, n_surrogate_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Points entering the loss, and the reference values on them.

    Mirrors ``bilevel.loss.masked_normalised_mse``: a point is kept if the reference
    is finite there and the normalised strain rate ``S*`` is below ``threshold``
    (inside the sheared region).  The surrogate may have been trained either on the
    full field or on the finite points only; both layouts are recognised from its
    number of points.

    Returns
    -------
    rows : ndarray of int
        Indices into the points of the surrogate.
    values : ndarray
        Reference values on those points.
    """
    reference = np.asarray(reference, dtype=float).ravel()
    s_star = np.asarray(s_star, dtype=float).ravel()
    finite = np.isfinite(reference)
    sheared = s_star[finite] < threshold
    values = reference[finite][sheared]
    if n_surrogate_points == reference.size:
        rows = np.flatnonzero(finite)[sheared]
    elif n_surrogate_points == int(finite.sum()):
        rows = np.flatnonzero(sheared)
    else:
        raise ValueError(
            f"the surrogate has {n_surrogate_points} points, which matches neither the "
            f"full reference field ({reference.size}) nor its finite part ({int(finite.sum())})"
        )
    return rows, values


def _baseline(data: Mapping[str, Any], case: str, variable: str) -> float:
    baseline = data["baseline"]
    if "loss" in baseline:
        baseline = baseline["loss"]
    value = float(baseline[case][variable])
    return value if value != 0.0 else 1.0


def likelihood_from_bilevel_data(
    data: Mapping[str, Any],
    cases: Sequence[str],
    variables: Sequence[str],
    *,
    s_star_threshold: float = 0.8,
    n_eff: float | Mapping[str, Mapping[str, float]] | None = None,
    noise: str = "marginal",
    sigma: float | Sequence[float] | None = None,
    validate: bool = True,
    n_validate: int = 5,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> FieldLikelihood:
    """Build the likelihood from the dictionaries of the deterministic framework.

    Each surrogate is first rebuilt as a batched :class:`PODEmulator` from the SMT
    ``PODI`` object it wraps (``surrogate.model`` or the surrogate itself).  The
    rebuilt emulator is checked against ``surrogate.predict`` on random points of
    the design box; if it does not reproduce it within ``tolerance`` -- or if the
    surrogate is not a ``PODI`` model -- ``surrogate.predict`` is used as is, which
    is always correct but slower.

    Parameters
    ----------
    data : mapping
        Dictionary with the keys ``"surrogates"``, ``"les"`` and ``"baseline"``.
    cases, variables : sequence of str
        Flow cases and quantities of interest entering the likelihood.
    s_star_threshold : float
        Threshold of the strain-rate sensor delimiting the sheared region.
    n_eff : float or mapping, optional
        Effective number of independent observations per field, either one value
        for all or ``{case: {variable: value}}``.  Defaults to the number of points
        in the mask; see :func:`sblcfd.cfd.effective_sample_size`.
    noise, sigma
        Error model, see :class:`FieldLikelihood`.
    validate : bool
        Check the rebuilt emulators against the original surrogates.
    n_validate : int
        Number of random points used for the check.
    tolerance : float
        Largest relative deviation accepted.
    seed : int

    Returns
    -------
    FieldLikelihood
        Its ``normalised_loss`` is the loss of the deterministic framework.
    """
    rng = np.random.default_rng(seed)
    terms = []
    for case in cases:
        s_star = data["les"][case]["S_star"]
        for variable in variables:
            surrogate = data["surrogates"][case][variable]
            reference = data["les"][case][variable]
            name = f"{case}/{variable}"

            def predict(x: np.ndarray, _s=surrogate) -> np.ndarray:
                return _as_rows(np.asarray(_s.predict(x), dtype=float), x.shape[0])

            emulator: PODEmulator | None = None
            podi = getattr(surrogate, "model", surrogate)
            if hasattr(podi, "interp_coeff") and hasattr(podi, "basis"):
                try:
                    emulator = PODEmulator.from_smt(podi)
                except (NotImplementedError, ValueError, AttributeError, KeyError) as error:
                    warnings.warn(
                        f"{name}: the surrogate could not be rebuilt ({error}); "
                        "falling back to surrogate.predict",
                        stacklevel=2,
                    )
            if emulator is not None and validate:
                lower, upper = emulator.bounds()
                points = lower + (upper - lower) * rng.random((n_validate, lower.size))
                deviation = emulator.validate(predict, points)
                if deviation > tolerance:
                    warnings.warn(
                        f"{name}: the rebuilt emulator deviates from surrogate.predict by "
                        f"{deviation:.2e} (relative); falling back to surrogate.predict",
                        stacklevel=2,
                    )
                    emulator = None

            if emulator is not None:
                n_points = emulator.n_points
            else:
                probe = np.atleast_2d(np.zeros(_input_dimension(surrogate, podi)))
                n_points = predict(np.vstack([probe, probe])).shape[1]
            rows, values = masked_rows(reference, s_star, s_star_threshold, n_points)
            # where the selected points sit in the full LES field, whatever the surrogate layout
            flat = np.asarray(reference, dtype=float).ravel()
            grid_index, _ = masked_rows(flat, s_star, s_star_threshold, flat.size)

            if isinstance(n_eff, Mapping):
                effective = n_eff[case][variable]
            else:
                effective = n_eff
            terms.append(
                FieldMisfit(
                    emulator if emulator is not None else predict,
                    values,
                    rows=rows,
                    name=name,
                    n_eff=effective,
                    scale=_baseline(data, case, variable),
                    grid_index=grid_index,
                )
            )
    return FieldLikelihood(terms, noise=noise, sigma=sigma)


def _input_dimension(surrogate: Any, podi: Any) -> int:
    """Number of inputs of a surrogate that could not be rebuilt."""
    for owner in (surrogate, podi):
        for attribute in ("n_inputs", "nx", "dim"):
            value = getattr(owner, attribute, None)
            if isinstance(value, (int, np.integer)):
                return int(value)
    models = getattr(podi, "interp_coeff", None)
    if models:
        return int(np.asarray(models[0].X_norma).shape[1])
    raise ValueError(
        "cannot infer the number of inputs of the surrogate; "
        "give it an integer attribute 'n_inputs'"
    )
