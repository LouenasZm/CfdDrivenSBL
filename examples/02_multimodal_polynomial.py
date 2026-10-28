"""Polynomial regression with a multimodal prior (Section 3.2 of the paper).

Data from ``y = 1 + x^2`` on a short interval are fitted with ``a0 + a1 x + a2 x^2``.
The intercept ``a0`` is a priori relevant and carries a three-mode prior; the slope
and curvature are questionable.  Three nested models explain the data almost equally
well (``2x``, ``1 + x^2`` and ``-1 + 4x - x^2``), so the posterior *and* the evidence
are multimodal: this is the case classical SBL cannot treat.

The model is linear, so ``likelihood x known prior`` is known in closed form (a
three-kernel Gaussian mixture).  The script runs NSBL on that exact mixture and on
the sampled one (TMCMC + kernel density estimate, as in the paper) and compares.

Run:  python examples/02_multimodal_polynomial.py [--figures DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sblcfd import NSBL, GammaHyperprior, GaussianMixture, GaussianMixturePrior, Parameter
from sblcfd.linear import linear_likelihood_mixture

NOISE_VARIANCE = 0.02


def main(seed: int = 3, figures: str | None = None) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.75, 1.25, 50)
    y = 1.0 + x**2 + rng.normal(0.0, np.sqrt(NOISE_VARIANCE), x.size)
    design = np.c_[np.ones_like(x), x, x**2]

    def log_likelihood(phi: np.ndarray) -> np.ndarray:
        residual = y[None, :] - phi @ design.T
        return -0.5 * (residual**2).sum(axis=1) / NOISE_VARIANCE - 0.5 * x.size * np.log(
            2 * np.pi * NOISE_VARIANCE
        )

    parameters = [
        Parameter("a0", prior=GaussianMixturePrior([-1.0, 0.0, 1.0], 0.2)),  # a priori relevant
        Parameter("a1", -10.0, 10.0),  # questionable
        Parameter("a2", -10.0, 10.0),  # questionable
    ]
    # the settings of the paper: near-Jeffreys Gamma hyperprior, applied to alpha itself
    model = NSBL(log_likelihood, parameters, hyperprior=GammaHyperprior.paper(), standardize=False)

    # exact mixture: the Gaussian likelihood times each of the three prior kernels
    likelihood = linear_likelihood_mixture(design, y, 1.0 / NOISE_VARIANCE)
    parts = [likelihood.multiply_gaussian([0], c, np.array([0.2**2])) for c in (-1.0, 0.0, 1.0)]
    exact = GaussianMixture(
        np.vstack([p.means for p in parts]),
        np.vstack([p.covariances for p in parts]),
        np.concatenate([p.log_weights for p in parts]) - np.log(3.0),
    )

    sampling = model.sample(2500, seed=seed)
    print(
        f"TMCMC: {sampling.n_evaluations:,} likelihood evaluations in {sampling.wall_time:.2f} s; "
        f"mass of likelihood x prior: sampled {model.log_mass_:.3f}, exact {exact.log_mass:.3f}"
    )
    sampled = model.fit_mixture("kde")

    print("\nNewton iteration from the two starting points of the paper:")
    print(
        f"  {'mixture':8s} {'start':>10s}   {'log(alpha_1)':>12s} {'log(alpha_2)':>12s}"
        "   kept   iterations"
    )
    for label, mixture in (("exact", exact), ("sampled", sampled)):
        model.set_mixture(mixture)
        for start in ([6.0, 8.0], [-3.0, -3.0]):
            result = model.optimize(starts=np.array([start]))
            kept = "+".join(n for n, k in zip(("a1", "a2"), result.relevant) if k)
            print(
                f"  {label:8s} {start!s:>10s}   {result.log_alpha[0]:12.2f} "
                f"{result.log_alpha[1]:12.2f}"
                f"   {kept:4s}   {result.optima[0].n_iterations}"
            )
    print("  paper     [6, 8]            -1.42         6.76   a1     (local optimum, y = 2x)")
    print("  paper   [-3, -3]             5.24         0.00   a2     (global optimum, y = 1 + x^2)")

    model.set_mixture(sampled)
    result = model.optimize(n_starts=16, seed=seed)
    print(f"\nMultistart on the sampled mixture: {len(result.optima)} distinct optima")
    for optimum in result.optima:
        kept = "+".join(n for n, g in zip(("a1", "a2"), optimum.relevance) if g > 0.5)
        print(
            f"  objective {optimum.objective:8.3f}   keeps {kept:5s}   "
            f"reached by {optimum.multiplicity} of 16 starts"
        )
    print()
    print(result.summary())

    if figures is not None:
        from sblcfd.plotting import plot_marginals, save_figure

        folder = Path(figures)
        folder.mkdir(parents=True, exist_ok=True)
        for mode in ("light", "dark"):
            fig = plot_marginals(result, sampled, truths=[1.0, 0.0, 1.0], mode=mode)
            save_figure(fig, folder / f"polynomial_marginals_{mode}.png")
        print(f"\nFigures written to {folder}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--figures", default=None, help="folder in which to save the figures")
    args = parser.parse_args()
    main(args.seed, args.figures)
