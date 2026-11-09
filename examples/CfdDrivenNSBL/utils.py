"""
Data loading and set-up for the NSBL version of the ERCOFTAC T3 example.

The surrogates, the LES fields, the baseline RANS fields and the S* mask are
loaded exactly as in ``examples/CfdDrivenOptim`` of the deterministic framework
(AugTransitionRANS, elastic net + bi-level optimisation), so that both
approaches are trained on identical inputs.  On top of that, this module builds
what NSBL needs: the effective number of observations of each field, the
parameters with their bounds and priors, and a check of the bounds against the
box covered by the surrogates.

The real data require (not needed with ``--toy``):
    - the high-fidelity database: https://doi.org/10.5281/zenodo.17166216
    - the postprocessing package: https://github.com/LouenasZm/ppMusicaa.git
    - the AugCfd package (``MusicaaSurrogate``, to unpickle the surrogates) and SMT

Author: Louenas Zemmour, October 2026
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sblcfd import Parameter, Uniform
from sblcfd.cfd import PODEmulator, effective_sample_size

# ============================================================================
#                          1.  Logging setup
# ============================================================================


def setup_logging(output_dir: Path) -> None:
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "nsbl.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info("Logging to %s", log_path)


# ============================================================================
#                          2.  Data loading (same as the deterministic example)
# ============================================================================


def load_surrogates(
    cases: list[str], variables: list[str], surrogate_paths: dict
) -> dict[str, dict[str, Any]]:
    """Load the pickled POD-Kriging surrogates, ``{case: {var: surrogate}}``."""
    try:  # the pickles refer to this class: it must be importable
        from AugCfd.surrogates.surrogates import MusicaaSurrogate  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "AugCfd is needed to unpickle the surrogates; copy it next to this script "
            "as in the deterministic example (see submit_slurm.sh)"
        ) from error

    surrogates: dict = {}
    for case in cases:
        surrogates[case] = {}
        for var in variables:
            path = surrogate_paths[case][var]
            with open(path, "rb") as f:
                surrogates[case][var] = pickle.load(f)
            logging.info("  Loaded surrogate  case=%-8s  var=%s  (%s)", case, var, path)
    return surrogates


def load_les(
    cases: list[str], variables: list[str], les_cfg: dict, baseline_cfg: dict
) -> dict[str, dict[str, np.ndarray]]:
    """LES fields interpolated on the RANS grid, with the S* field (``"S_star"``)."""
    from ppModule.interface import PostProcessMusicaa
    from preprocess import ErcoftacPreprocess

    les: dict = {}
    for case in cases:
        pp = PostProcessMusicaa(config=les_cfg[case])
        pp.compute_qty(qty="tauw")
        stats = pp.return_stats()
        pp_obj = ErcoftacPreprocess(case=case, config=baseline_cfg)
        # preprocess_les appends to the key list: pass a copy
        les[case] = pp_obj.preprocess_les(
            grid=pp.config["grid"], stats=stats, keylist=list(variables)
        )
        logging.info("  Loaded LES data   case=%s", case)
    return les


def load_baseline(
    cases: list[str], variables: list[str], baseline_cfg: dict, les: dict
) -> tuple[dict, dict]:
    """Baseline RANS fields and their MSE against the LES.

    The MSE is computed as in ``compute_baseline_losses`` of the deterministic
    example (all finite LES points, no S* mask), so that the normalised loss
    reported here is the one minimised by the elastic net.  The fields themselves
    are kept as well: their residual against the LES gives the effective number of
    independent observations.
    """
    from ppModule.interface import PostProcessMusicaa
    from preprocess import ErcoftacPreprocess

    fields: dict = {}
    losses: dict = {}
    for case in cases:
        pp = PostProcessMusicaa(config=baseline_cfg[case])
        pp.compute_qty(qty="tauw")
        stats = pp.return_stats()
        pp_obj = ErcoftacPreprocess(case=case, config=baseline_cfg)
        fields[case] = pp_obj.preprocess_rans(
            grid=pp.config["grid"], stats=stats, keylist=list(variables)
        )
        losses[case] = {}
        for var in variables:
            reference = _as_array(les[case][var])
            finite = np.isfinite(reference)
            losses[case][var] = float(
                np.mean((_as_array(fields[case][var])[finite] - reference[finite]) ** 2)
            )
        logging.info("  Loaded baseline   case=%s", case)
    return fields, losses


def build_data(config: dict, cases: list[str]) -> dict:
    """Assemble the data dictionary for ``cases``.

    Same layout as the ``follower_data`` of the deterministic example, plus the
    baseline fields::

        {"surrogates":      {case: {var: surrogate}},
         "les":             {case: {var: field, "S_star": field}},
         "baseline":        {"loss": {case: {var: float}}},
         "baseline_fields": {case: {var: field}}}
    """
    variables = config["variables"]
    logging.info("── Loading data (%s) ──", cases)
    les = load_les(cases, variables, config["les"], config["baseline"])
    surrogates = load_surrogates(cases, variables, config["paths"]["surrogates"])
    fields, losses = load_baseline(cases, variables, config["baseline"], les)
    return {
        "surrogates": surrogates,
        "les": les,
        "baseline": {"loss": losses},
        "baseline_fields": fields,
    }


def subset(data: dict, cases: list[str]) -> dict:
    """The part of ``data`` that concerns ``cases``."""
    return {
        "surrogates": {c: data["surrogates"][c] for c in cases},
        "les": {c: data["les"][c] for c in cases},
        "baseline": {"loss": {c: data["baseline"]["loss"][c] for c in cases}},
        "baseline_fields": {c: data["baseline_fields"][c] for c in cases},
    }


# ============================================================================
#                          3.  What NSBL adds
# ============================================================================


def _as_array(field: Any) -> np.ndarray:
    """Plain float array, masked entries replaced by NaN."""
    return np.asarray(np.ma.filled(np.ma.asarray(field, dtype=float), np.nan), dtype=float)


def sheared_region(les_case: dict, var: str, threshold: float) -> np.ndarray:
    """Points entering the loss: finite LES value and S* below the threshold."""
    reference = _as_array(les_case[var])
    s_star = _as_array(les_case["S_star"])
    with np.errstate(invalid="ignore"):
        return np.isfinite(reference) & (s_star < threshold)


def baseline_diagnostics(
    data: dict, cases: list[str], variables: list[str], threshold: float
) -> dict[str, dict[str, dict[str, float]]]:
    """Effective number of observations and masked MSE of the baseline, per field.

    The residual ``baseline RANS - LES`` on the sheared region is spatially
    correlated: neighbouring mesh points carry nearly the same information.
    ``effective_sample_size`` turns its autocorrelation along both grid directions
    into an equivalent number of independent observations.  The baseline residual
    is the natural choice before any model is identified, and a conservative one:
    its correlation length is larger than that of a corrected model.
    """
    out: dict = {}
    for case in cases:
        out[case] = {}
        for var in variables:
            region = sheared_region(data["les"][case], var, threshold)
            residual = _as_array(data["baseline_fields"][case][var]) - _as_array(
                data["les"][case][var]
            )
            if residual.shape != region.shape:
                raise ValueError(
                    f"{case}/{var}: the baseline field {residual.shape} and the LES field "
                    f"{region.shape} are not on the same grid"
                )
            residual = np.where(region & np.isfinite(residual), residual, np.nan)
            out[case][var] = {
                "n_points": int(np.isfinite(residual).sum()),
                "n_eff": effective_sample_size(residual),
                "mse_masked": float(np.nanmean(residual**2)),
            }
    return out


def resolve_n_eff(setting: Any, diagnostics: dict, likelihood) -> dict:
    """Effective number of observations of every field, from the ``n_eff`` setting.

    ``"baseline"`` -- from the baseline residual (default);
    ``"points"``   -- every mesh point counts as an independent observation
                      (over-confident: keeps nearly every term);
    a number       -- the same value for every field;
    a dict         -- ``{case: {var: value}}``.
    """
    n_eff: dict = {}
    for term in likelihood.terms:
        case, var = term.name.split("/")
        if setting == "baseline":
            value = diagnostics[case][var]["n_eff"]
        elif setting == "points":
            value = term.n_points
        elif isinstance(setting, (int, float)):
            value = float(setting)
        else:
            value = float(setting[case][var])
        n_eff.setdefault(case, {})[var] = float(np.clip(value, 1.0, term.n_points))
    return n_eff


def make_parameters(
    names: list[str], lower: list[float], upper: list[float], a_priori_relevant: list
) -> list[Parameter]:
    """Questionable parameters (ARD prior), except those listed as a priori relevant.

    An a priori relevant coefficient (given by name or by 0-based index) receives a
    uniform prior on its bounds and is never pruned.
    """
    relevant = set()
    for item in a_priori_relevant:
        relevant.add(names.index(item) if isinstance(item, str) else int(item))
    parameters = []
    for i, (name, lo, hi) in enumerate(zip(names, lower, upper)):
        if i in relevant:
            parameters.append(Parameter(name, lo, hi, prior=Uniform(lo, hi)))
        else:
            parameters.append(Parameter(name, lo, hi))
    return parameters


def coefficient_names(config: dict) -> list[str]:
    """Names from the config, or ``Gamma_01 ... Gamma_21``."""
    n = config["coefficients"]["n"]
    names = config["coefficients"].get("names")
    if names is None:
        return [f"Gamma_{i + 1:02d}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"coefficients.names has {len(names)} entries, expected {n}")
    return list(names)


def surrogate_box(likelihood) -> tuple[np.ndarray, np.ndarray] | None:
    """Box covered by the designs of experiments of all the surrogates (intersection)."""
    lower, upper = None, None
    for term in likelihood.terms:
        if isinstance(term.emulator, PODEmulator):
            lo, hi = term.emulator.bounds()
            lower = lo if lower is None else np.maximum(lower, lo)
            upper = hi if upper is None else np.minimum(upper, hi)
    if lower is None:
        return None
    return lower, upper


def check_bounds(names: list[str], lower, upper, box) -> list[str]:
    """Compare the bounds of the coefficients with the box covered by the surrogates.

    Outside the design of experiments a Kriging model returns to its trend and its
    predictions mean nothing, so the bounds must lie inside the box.  A coefficient
    whose lower bound is zero is sign-constrained (half-normal ARD prior): a
    mismatch in the *sign* of the bounds therefore changes the model, not only the
    range.  Returns the list of warnings, also written to the log.
    """
    warnings_: list[str] = []
    if box is None:
        logging.info("  Surrogate box unknown (surrogates not rebuilt): bounds not checked")
        return warnings_
    lo_box, hi_box = box
    w = max(12, max(len(name) for name in names))
    logging.info(
        "  %-*s %10s %10s   %10s %10s", w, "coefficient", "lower", "upper", "DOE min", "DOE max"
    )
    for i, name in enumerate(names):
        width = max(hi_box[i] - lo_box[i], 1e-12)
        flag = ""
        if lower[i] < lo_box[i] - 0.02 * width or upper[i] > hi_box[i] + 0.02 * width:
            flag = "  <-- outside the surrogate box"
            warnings_.append(
                f"{name}: bounds [{lower[i]:g}, {upper[i]:g}] exceed the design of experiments "
                f"[{lo_box[i]:.3g}, {hi_box[i]:.3g}]"
            )
        logging.info(
            "  %-*s %10.3g %10.3g   %10.3g %10.3g%s",
            w,
            name,
            lower[i],
            upper[i],
            lo_box[i],
            hi_box[i],
            flag,
        )
    for message in warnings_:
        logging.warning(message)
    return warnings_


def load_reference_models(entries: dict, n: int) -> dict[str, np.ndarray]:
    """Coefficient vectors to compare with, e.g. the bi-level model M_BL.

    Each entry is either a list of ``n`` values or the path to a results file of
    the deterministic framework (``bilevel_results.json``, key ``"coefficients"``),
    which stores the coefficients in the layout of the surrogates.
    """
    models: dict[str, np.ndarray] = {}
    for name, entry in (entries or {}).items():
        if isinstance(entry, str):
            with open(entry, encoding="utf-8") as f:
                entry = json.load(f)["coefficients"]
        vector = np.asarray(entry, dtype=float).ravel()
        if vector.size != n:
            raise ValueError(f"reference model '{name}' has {vector.size} coefficients, not {n}")
        models[name] = vector
    return models
