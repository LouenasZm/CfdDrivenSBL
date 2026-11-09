"""Template: NSBL on the surrogates and LES data of the deterministic framework.

NOT runnable as shipped: it needs the surrogates, the LES database and the helper
modules of the elastic-net / bi-level framework (AugTransitionRANS).  It shows the
few lines that turn the dictionaries built there into a sparse Bayesian study.

    python examples/05_from_bilevel_framework.py config.json \
        --bilevel-example /path/to/AugTransitionRANS/examples/CfdDrivenOptim

``config.json`` is the configuration file of that framework: the training cases,
the variables, the bounds of the coefficients and the paths are read from it.

What to decide before trusting the result
-----------------------------------------
* ``n_eff``.  The number of independent observations in each field.  The default
  (every mesh point of the mask) treats the points as independent and gives a far
  too narrow posterior.  Estimate it from a residual field with
  ``sblcfd.cfd.effective_sample_size`` (step 2 below) or set it by hand.
* ``truncate_prior=True``.  The bounds are the box covered by the design of
  experiments: outside it the Kriging models return to their trend and mean nothing.
* Which terms are questionable.  Here all of them; give a ``prior`` to a
  ``Parameter`` to declare a coefficient a priori relevant instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sblcfd import NSBL, Parameter, model_average
from sblcfd.cfd import effective_sample_size, likelihood_from_bilevel_data


def main(config_path: str, bilevel_example: str, n_particles: int, output: str) -> None:
    sys.path.insert(0, bilevel_example)
    from utils import build_data_dicts  # noqa: PLC0415  (module of the deterministic framework)

    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)
    cases, variables = config["cases"]["training"], config["variables"]
    threshold = config.get("s_star_threshold", 0.8)

    # -- 1. same surrogates, same LES fields, same mask as the elastic-net loss ------ #
    _, data = build_data_dicts(config)
    likelihood = likelihood_from_bilevel_data(data, cases, variables, s_star_threshold=threshold)

    # the two approaches now evaluate the same misfit; compare with compute_normalised_loss
    baseline = np.zeros((1, config["coefficients"]["n"]))
    print(f"normalised loss of the baseline model: {likelihood.normalised_loss(baseline)[0]:.4f}")

    # -- 2. effective number of observations, from the residual of the baseline ------ #
    for term in likelihood.terms:
        case, variable = term.name.split("/")
        shape = np.asarray(data["les"][case][variable]).shape
        if len(shape) == 2:  # structured (n_x, n_y) field: correlation along both directions
            term.n_eff = effective_sample_size(term.residual_field(baseline, shape))
        print(f"  {term.name:14s} {term.n_points:6d} points, n_eff = {term.n_eff:.1f}")

    # -- 3. parameters: every coefficient is questionable, within its bounds ---------- #
    lower, upper = config["coefficients"]["l_bounds"], config["coefficients"]["u_bounds"]
    parameters = [
        Parameter(f"Gamma_{i + 1}", lo, hi) for i, (lo, hi) in enumerate(zip(lower, upper))
    ]

    # -- 4. sparse Bayesian learning ------------------------------------------------- #
    model = NSBL(likelihood, parameters, truncate_prior=True)
    sampling = model.sample(n_particles, seed=config.get("seed", 1), verbose=True)
    model.fit_mixture()
    result = model.optimize(n_starts=32, seed=config.get("seed", 1))
    print(result.summary())

    # -- 5. candidates ranked by sampled evidence, then model averaging --------------- #
    ranked = model.compare_structures(n_candidates=5, n_particles=n_particles // 2, seed=0)
    best = ranked[0]
    print(best.summary())
    print(
        f"log Bayes factor against the full model: {best.log_evidence - sampling.log_evidence:+.2f}"
    )

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "nsbl_result.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                **result.to_dict(),
                "selected_structure": best.active.tolist(),
                "coefficients_mean": best.mean().tolist(),
                "coefficients_std": best.std().tolist(),
                "coefficients_best": best.best().tolist(),
                "log_evidence_sparse": best.log_evidence,
                "log_evidence_full": sampling.log_evidence,
            },
            stream,
            indent=2,
        )
    # coefficient vectors to propagate through the RANS solver for predictive bands
    np.savetxt(folder / "posterior_draws.txt", model_average(ranked, 100, seed=0))
    print(f"results written to {folder}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="configuration file of the bi-level framework")
    parser.add_argument("--bilevel-example", required=True, help="folder containing utils.py")
    parser.add_argument("--particles", type=int, default=4000)
    parser.add_argument("--output", default="results_nsbl")
    args = parser.parse_args()
    main(args.config, args.bilevel_example, args.particles, args.output)
