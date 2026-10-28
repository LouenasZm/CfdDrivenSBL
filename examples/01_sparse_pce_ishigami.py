"""Sparse polynomial-chaos surrogate of the Ishigami function (Section 3.1 of the paper).

A linear-in-parameter problem, where NSBL must agree with classical sparse Bayesian
learning (SBL): the likelihood is exactly one Gaussian kernel, so nothing is
approximated.  The two differ only in how the evidence is maximised: SBL updates the
precisions by fixed-point sweeps, NSBL moves all of them at once with Newton steps
that use the exact Hessian.

Run:  python examples/01_sparse_pce_ishigami.py
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
from numpy.polynomial import legendre
from scipy.stats import qmc

from sblcfd import GammaHyperprior
from sblcfd.linear import nsbl_linear, sbl_linear


def ishigami(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def legendre_basis(xi: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal Legendre chaos of total degree ``order`` for germs uniform on [-1, 1]."""
    dim = xi.shape[1]
    degrees = np.array([m for m in product(range(order + 1), repeat=dim) if sum(m) <= order])
    degrees = degrees[np.lexsort(degrees.T[::-1])][np.argsort(degrees.sum(axis=1), kind="stable")]
    univariate = np.stack(
        [
            legendre.legvander(xi[:, j], order) * np.sqrt(2 * np.arange(order + 1) + 1)
            for j in range(dim)
        ]
    )  # (dim, n_samples, order + 1)
    basis = np.prod([univariate[j][:, degrees[:, j]] for j in range(dim)], axis=0)
    return basis, degrees


def sobol_indices(coefficients: np.ndarray, degrees: np.ndarray) -> dict[str, float]:
    """First-order and total Sobol indices read off the chaos coefficients."""
    variance = np.sum(coefficients[1:] ** 2)
    out = {}
    for j in range(degrees.shape[1]):
        only_j = (degrees[:, j] > 0) & (np.delete(degrees, j, axis=1).sum(axis=1) == 0)
        out[f"S{j + 1}"] = float(np.sum(coefficients[only_j] ** 2) / variance)
        out[f"ST{j + 1}"] = float(np.sum(coefficients[degrees[:, j] > 0] ** 2) / variance)
    return out


def main(order: int = 6, n_samples: int = 250, seed: int = 0) -> None:
    xi = 2.0 * qmc.LatinHypercube(d=3, seed=seed).random(n_samples) - 1.0
    data = ishigami(np.pi * xi)
    basis, degrees = legendre_basis(xi, order)
    print(
        f"Ishigami function, {n_samples} samples, chaos of order {order}: {basis.shape[1]} terms\n"
    )

    clock = time.perf_counter()
    classic = sbl_linear(basis, data, hyperprior=GammaHyperprior(1e-5, 1e-5))
    t_classic = time.perf_counter() - clock

    # start from an (almost) empty model: only the mean term is active
    start = np.full(basis.shape[1], 5.0)
    start[0] = 0.0
    clock = time.perf_counter()
    newton = nsbl_linear(basis, data, hyperprior=GammaHyperprior(1e-5, 1e-5), theta0=start)
    t_newton = time.perf_counter() - clock

    print(f"{'':28s}{'SBL (fixed point)':>20s}{'NSBL (Newton)':>20s}")
    rows = [
        (
            "terms kept (relevance > 0.5)",
            int((classic.relevance > 0.5).sum()),
            int((newton.relevance > 0.5).sum()),
        ),
        ("log-evidence", f"{classic.log_evidence:.2f}", f"{newton.log_evidence:.2f}"),
        ("noise variance", f"{1 / classic.precision:.2e}", f"{1 / newton.precision:.2e}"),
        ("evidence evaluations", classic.n_evidence_evaluations, newton.n_evidence_evaluations),
        ("wall time [s]", f"{t_classic:.2f}", f"{t_newton:.2f}"),
    ]
    for label, a, b in rows:
        print(f"{label:28s}{a!s:>20s}{b!s:>20s}")
    same = np.array_equal(classic.relevance > 0.5, newton.relevance > 0.5)
    print(f"\nSame set of terms: {same}; largest difference between the coefficients: "
          f"{np.abs(classic.mean - newton.mean).max():.1e}")  # fmt: skip

    exact = {"S1": 0.3139, "S2": 0.4424, "S3": 0.0, "ST1": 0.5576, "ST2": 0.4424, "ST3": 0.2437}
    estimate = sobol_indices(newton.mean, degrees)
    print("\nSobol indices from the sparse surrogate (analytical value in brackets):")
    print("  " + "   ".join(f"{k} = {estimate[k]:.3f} ({v:.3f})" for k, v in exact.items()))


if __name__ == "__main__":
    main()
