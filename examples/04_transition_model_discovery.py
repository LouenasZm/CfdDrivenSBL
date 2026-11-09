"""CFD-driven discovery of a sparse correction of a transition model -- in miniature.

This is the workflow of a real application with the RANS solver replaced by a toy
model that runs in milliseconds (``sblcfd.cfd.ToyTransition``).  Everything else is
what you would do with MUSICAA and SMT surrogates:

1. a dictionary of 21 candidate terms correcting the production and destruction of
   intermittency, with sign constraints on the coefficients of the exponentials;
2. a design of experiments and one POD-Kriging emulator per (flow case, quantity);
3. reference fields for the training cases (here generated from a hidden sparse
   model, plus spatially correlated noise);
4. a likelihood on the boundary-layer mask, tempered for spatial correlation;
5. NSBL: sample once, fit a Gaussian mixture, maximise the evidence;
6. the candidate structures re-ranked by their sampled evidence;
7. the selected model pushed through the *solver* on training and unseen cases.

Run:  python examples/04_transition_model_discovery.py [--figures DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import qmc

from sblcfd import NSBL, model_average
from sblcfd.cfd import (
    FieldLikelihood,
    FieldMisfit,
    PODEmulator,
    effective_sample_size,
    toy_cases,
    transition_library,
)

TRAIN = ["zpg-low", "zpg-high", "pg-mixed"]
TEST = ["pg-adverse", "pg-favourable"]
QUANTITIES = ["u", "dudy"]


def main(n_particles: int = 4000, n_design: int = 512, seed: int = 0, figures: str | None = None):
    rng = np.random.default_rng(seed)
    clock = time.perf_counter()

    # -- 1. dictionary and flow cases ---------------------------------------- #
    library = transition_library()
    parameters = (
        library.parameters()
    )  # constants in [0, 1], lam-terms in [-1, 1], exponents in [0, 1]
    lower = np.array([p.lower for p in parameters])
    upper = np.array([p.upper for p in parameters])
    cases = toy_cases()
    baseline = np.zeros(library.n_parameters)  # all coefficients zero: the uncorrected model

    # the model that generated the reference data; the inference never sees it
    hidden = np.zeros(library.n_parameters)
    hidden[[0, 1, 4, 11, 13, 15]] = [0.25, -0.35, 0.9, 0.5, 0.4, 0.8]
    print("Hidden model behind the reference data:")
    print("  " + library.format(hidden).replace("\n", "\n  "))

    # -- 2. design of experiments and emulators -------------------------------- #
    design = lower + (upper - lower) * qmc.Sobol(library.n_parameters, seed=seed).random(n_design)
    terms = []
    print(f"\nEmulators ({n_design} solver runs per case):")
    for name in TRAIN:
        case = cases[name]
        snapshots = case.fields(design)  # the "CFD runs"
        truth = case.fields(hidden)
        uncorrected = case.fields(baseline)
        mask = case.boundary_layer_mask(hidden)  # the role played by the S* sensor
        for quantity in QUANTITIES:
            emulator = PODEmulator.fit(design, snapshots[quantity], energy=0.9999, seed=seed)

            # -- 3. reference data: hidden model + correlated noise, three times
            #       smaller than what the baseline model misses
            signal = np.sqrt(np.mean((truth[quantity][0] - uncorrected[quantity][0])[mask] ** 2))
            noise = gaussian_filter(rng.standard_normal(case.shape), (3.0, 1.5)).ravel()
            reference = truth[quantity][0] + noise * signal / 3.0 / noise[mask].std()

            # -- 4. misfit on the mask; n_eff from the residual of the baseline
            term = FieldMisfit(
                emulator, reference, rows=np.flatnonzero(mask), name=f"{name}/{quantity}"
            )
            term.scale = float(term.mse(baseline)[0])  # baseline error, for reporting
            term.n_eff = effective_sample_size(term.residual_field(baseline, case.shape))
            terms.append(term)
            print(
                f"  {term.name:15s} {emulator.n_modes} POD modes, {term.n_points} points in the "
                f"mask, {term.n_eff:5.1f} effective observations"
            )
    likelihood = FieldLikelihood(terms)  # error variances integrated out analytically
    t_emulators = time.perf_counter() - clock

    # -- 5. nonlinear sparse Bayesian learning ----------------------------------- #
    # truncate_prior=True: the box is a genuine constraint (the range covered by the
    # emulators), so the ARD priors are Gaussians truncated to it.
    model = NSBL(likelihood, parameters, truncate_prior=True)

    clock = time.perf_counter()
    sampling = model.sample(n_particles, seed=seed)
    t_sampling = time.perf_counter() - clock
    print(
        f"\nSampling: {sampling.n_evaluations:,} likelihood evaluations in {t_sampling:.1f} s "
        f"({1e6 * t_sampling / sampling.n_evaluations:.0f} us each), {sampling.n_stages} stages; "
        f"log-evidence of the full model {sampling.log_evidence:.2f} "
        f"+- {sampling.log_evidence_error:.2f}"
    )

    clock = time.perf_counter()
    mixture = model.fit_mixture(seed=seed)  # "auto": EM in 21 dimensions
    result = model.optimize(n_starts=32, seed=seed)
    t_learning = time.perf_counter() - clock
    print(
        f"Mixture ({mixture.n_components} kernels) and 32 Newton runs on the evidence: "
        f"{t_learning:.1f} s\n"
    )
    print(result.summary())
    if len(result.optima) == 1:
        print("\nAll 32 Newton runs converge to this structure.")
    else:
        probability = result.inclusion_probability()
        ambiguous = [
            result.names[i] for i, p in zip(result.ard_index, probability) if 0.05 < p < 0.95
        ]
        print(f"\nTerms that only some of the {len(result.optima)} optima keep: {ambiguous}")

    # -- 6. candidate structures ranked without the mixture approximation ---------- #
    clock = time.perf_counter()
    ranked = model.compare_structures(n_candidates=5, n_particles=2000, seed=seed + 1)
    t_ranking = time.perf_counter() - clock
    print(f"\nCandidate structures, ranked by sampled evidence ({t_ranking:.1f} s):")
    for calibration in ranked:
        kept = ", ".join(n for n, a in zip(calibration.names, calibration.active) if a)
        print(
            f"  log-evidence {calibration.log_evidence:8.2f} "
            f"+- {calibration.log_evidence_error:.2f}"
            f"   {kept}"
        )
    best = ranked[0]
    print(
        f"  log Bayes factor of the best structure against the full 21-term model: "
        f"{best.log_evidence - sampling.log_evidence:+.2f}\n"
    )
    print(best.summary())
    print("\nIdentified model (posterior mean ± standard deviation):")
    print(
        "  " + library.format(best.mean(), active=best.active, std=best.std()).replace("\n", "\n  ")
    )
    improvement = likelihood.improvement(best.mean()[None])[0]
    print("\nImprovement over the baseline on the training fields (1 = perfect):")
    for term, score in zip(terms, improvement):
        print(f"  {term.name:15s} {100 * score:5.1f} %")

    # -- 7. a posteriori: through the solver, on training and unseen cases ----------- #
    # averaging over the candidate structures carries the uncertainty on the structure
    draws = model_average(ranked, 300, seed=seed)
    print("\nSkin friction through the solver (relative rms error against the hidden model):")
    panels = {}
    for name in TRAIN + TEST:
        case = cases[name]
        exact = case.skin_friction(hidden)[0]
        uncorrected = case.skin_friction(baseline)[0]
        predicted = case.skin_friction(draws)
        scale = np.sqrt(np.mean(exact**2))
        before = np.sqrt(np.mean((uncorrected - exact) ** 2)) / scale
        after = np.sqrt(np.mean((predicted.mean(axis=0) - exact) ** 2)) / scale
        tag = "training" if name in TRAIN else "unseen"
        print(
            f"  {name:14s} ({tag:8s})  baseline {100 * before:5.1f} %  ->  "
            f"sparse model {100 * after:4.1f} %"
        )
        panels[f"{name} ({tag})"] = {
            "x": case.x,
            "draws": 1e3 * predicted,
            "reference": 1e3 * exact,
            "baseline": 1e3 * uncorrected,
        }  # fmt: skip

    print(
        f"\nWall time: emulators {t_emulators:.1f} s, sampling {t_sampling:.1f} s, "
        f"sparse learning {t_learning:.1f} s, ranking {t_ranking:.1f} s"
    )

    if figures is not None:
        from sblcfd.plotting import plot_predictions, plot_relevance, save_figure

        folder = Path(figures)
        folder.mkdir(parents=True, exist_ok=True)
        for mode in ("light", "dark"):
            save_figure(
                plot_relevance(result, mode=mode), folder / f"transition_relevance_{mode}.png"
            )
            fig = plot_predictions(
                panels, xlabel="x / L", ylabel=r"$C_f \times 10^3$", mode=mode,
                reference_label="hidden model",
            )  # fmt: skip
            save_figure(fig, folder / f"transition_predictions_{mode}.png")
        print(f"Figures written to {folder}/")
    return result, ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--particles", type=int, default=4000, help="population of the sampler")
    parser.add_argument("--design", type=int, default=512, help="solver runs per flow case")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figures", default=None, help="folder in which to save the figures")
    args = parser.parse_args()
    main(args.particles, args.design, args.seed, args.figures)
