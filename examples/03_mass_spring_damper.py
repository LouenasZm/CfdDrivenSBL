"""Sparse damping of a three-storey shear frame (Section 3.3 of the paper).

A three-degree-of-freedom mass-spring-damper chain vibrates freely; only the
displacement of the top floor is observed, with noise.  Damping truly acts between
the ground and the first floor only.  The stiffnesses are a priori relevant (uniform
priors), the three damping coefficients are questionable.  The response depends
nonlinearly on all six, so the posterior is far from Gaussian.

NSBL finds that ``c2`` and ``c3`` are not needed.  The script then calibrates the
sparse model directly and compares its evidence with that of the full model.

Run:  python examples/03_mass_spring_damper.py
"""

from __future__ import annotations

import numpy as np

from sblcfd import NSBL, GammaHyperprior, Parameter, Uniform

TIMES = np.linspace(0.0, 4.0, 100)
INITIAL_STATE = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # second floor displaced, at rest
# The paper quotes a noise "variance of 0.1", which leaves the damping distribution
# ambiguous with these 100 observations (several modes, the selection depends on the
# noise realisation).  A standard deviation of 0.05 gives the single-mode posterior
# shown in its Fig. 10.
NOISE_STD = 0.05


def top_floor_displacement(phi: np.ndarray) -> np.ndarray:
    """Displacement of the third floor for a batch of ``(c1, c2, c3, k1, k2, k3)``.

    Unit masses; the free response of ``x' = A x`` is evaluated for all parameter
    vectors at once through a batched eigendecomposition.
    """
    phi = np.atleast_2d(phi)
    n = phi.shape[0]

    def chain(v: np.ndarray) -> np.ndarray:
        m = np.zeros((n, 3, 3))
        m[:, 0, 0], m[:, 1, 1], m[:, 2, 2] = v[:, 0] + v[:, 1], v[:, 1] + v[:, 2], v[:, 2]
        m[:, 0, 1] = m[:, 1, 0] = -v[:, 1]
        m[:, 1, 2] = m[:, 2, 1] = -v[:, 2]
        return m

    a = np.zeros((n, 6, 6))
    a[:, :3, 3:] = np.eye(3)
    a[:, 3:, :3] = -chain(phi[:, 3:])
    a[:, 3:, 3:] = -chain(phi[:, :3])
    values, vectors = np.linalg.eig(a)
    amplitudes = np.linalg.solve(
        vectors, np.broadcast_to(INITIAL_STATE.astype(complex), (n, 6))[..., None]
    )
    with np.errstate(
        over="ignore", invalid="ignore"
    ):  # negative damping makes the response blow up
        modes = np.exp(values[:, :, None] * TIMES[None, None, :])
        return np.real(np.einsum("nm,nmt->nt", vectors[:, 2, :] * amplitudes[..., 0], modes))


def main(seed: int = 0) -> None:
    rng = np.random.default_rng(5)
    truth = np.array([10.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0])
    data = top_floor_displacement(truth)[0] + rng.normal(0.0, NOISE_STD, TIMES.size)

    def log_likelihood(phi: np.ndarray) -> np.ndarray:
        residual = top_floor_displacement(phi) - data
        return -0.5 * (residual**2).sum(axis=1) / NOISE_STD**2 - TIMES.size * np.log(
            NOISE_STD * np.sqrt(2 * np.pi)
        )

    parameters = [Parameter(f"c{i}", -20.0, 40.0) for i in (1, 2, 3)]  # questionable
    parameters += [
        Parameter(f"k{i}", prior=Uniform(0.0, 5000.0)) for i in (1, 2, 3)
    ]  # known priors
    model = NSBL(log_likelihood, parameters, hyperprior=GammaHyperprior.paper(), standardize=False)

    sampling = model.sample(2500, seed=seed, max_steps=20)
    print(
        f"TMCMC on the full model: {sampling.n_evaluations:,} likelihood evaluations in "
        f"{sampling.wall_time:.1f} s, log-evidence {sampling.log_evidence:.2f} "
        f"+- {sampling.log_evidence_error:.2f}"
    )
    mean, spread = sampling.samples.mean(axis=0), sampling.samples.std(axis=0)
    print("  with flat priors on the damping coefficients (mean ± standard deviation):")
    print(
        "    "
        + ",  ".join(f"{p.name} = {m:.4g} ± {s:.3g}" for p, m, s in zip(parameters, mean, spread))
    )

    model.fit_mixture("kde")
    print("\nNewton iteration from the three starting points of the paper:")
    for start in (-5.0, 0.0, 5.0):
        result = model.optimize(starts=np.full((1, 3), start))
        values = ", ".join(f"{v:6.2f}" for v in result.log_alpha)
        print(
            f"  start {start:4.1f}:  log(alpha) = [{values}]   "
            f"({result.optima[0].n_iterations} iterations)"
        )
    print("  paper      :  log(alpha) = [ -5.02,   4.03,   4.42]")

    result = model.optimize(n_starts=16, seed=seed)
    print()
    print(result.summary())

    calibration = model.calibrate(n_particles=2500, seed=seed + 1, max_steps=20)
    print()
    print(calibration.summary())
    print(
        f"\nLog Bayes factor of the sparse model against the full one: "
        f"{calibration.log_evidence - sampling.log_evidence:+.2f}"
    )
    print("True values: c1 = 10, k1 = k2 = k3 = 1000")


if __name__ == "__main__":
    main()
