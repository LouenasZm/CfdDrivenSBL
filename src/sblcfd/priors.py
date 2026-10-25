"""Prior distributions of the a priori relevant parameters.

Only the *known* priors live here.  The questionable parameters carry Gaussian ARD
priors whose precisions are learnt, and those are handled analytically in
:mod:`sblcfd.evidence`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp

__all__ = ["GaussianMixturePrior", "HalfNormal", "JointPrior", "Normal", "Prior", "Uniform"]

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


class Prior:
    """One-dimensional prior.  Subclasses define ``lower``, ``upper``, ``logpdf`` and ``sample``."""

    lower: float = -np.inf
    upper: float = np.inf

    def logpdf(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Uniform(Prior):
    """Uniform density on ``[lower, upper]``."""

    def __init__(self, lower: float, upper: float) -> None:
        if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
            raise ValueError("a uniform prior needs finite bounds with upper > lower")
        self.lower, self.upper = float(lower), float(upper)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        inside = (x >= self.lower) & (x <= self.upper)
        return np.where(inside, -np.log(self.upper - self.lower), -np.inf)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower, self.upper, n)

    def __repr__(self) -> str:
        return f"Uniform({self.lower:g}, {self.upper:g})"


class Normal(Prior):
    """Gaussian density with the given mean and standard deviation."""

    def __init__(self, mean: float, std: float) -> None:
        if std <= 0.0:
            raise ValueError("the standard deviation must be positive")
        self.mean, self.std = float(mean), float(std)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self.mean) / self.std
        return -0.5 * z * z - np.log(self.std) - _LOG_SQRT_2PI

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, n)

    def __repr__(self) -> str:
        return f"Normal({self.mean:g}, {self.std:g})"


class HalfNormal(Prior):
    """Zero-mode Gaussian restricted to the non-negative half line (e.g. a noise scale)."""

    lower = 0.0

    def __init__(self, std: float) -> None:
        if std <= 0.0:
            raise ValueError("the scale must be positive")
        self.std = float(std)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = x / self.std
        value = np.log(2.0) - 0.5 * z * z - np.log(self.std) - _LOG_SQRT_2PI
        return np.where(x >= 0.0, value, -np.inf)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.abs(rng.normal(0.0, self.std, n))

    def __repr__(self) -> str:
        return f"HalfNormal({self.std:g})"


class GaussianMixturePrior(Prior):
    """One-dimensional Gaussian mixture, for multimodal prior knowledge."""

    def __init__(
        self,
        means: Sequence[float],
        stds: Sequence[float] | float,
        weights: Sequence[float] | None = None,
    ) -> None:
        self.means = np.asarray(means, dtype=float)
        self.stds = np.broadcast_to(np.asarray(stds, dtype=float), self.means.shape).copy()
        w = np.ones_like(self.means) if weights is None else np.asarray(weights, dtype=float)
        self.weights = w / w.sum()

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=float)[..., None] - self.means) / self.stds
        terms = np.log(self.weights) - 0.5 * z * z - np.log(self.stds) - _LOG_SQRT_2PI
        return logsumexp(terms, axis=-1)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        component = rng.choice(self.means.size, size=n, p=self.weights)
        return rng.normal(self.means[component], self.stds[component])

    def __repr__(self) -> str:
        return f"GaussianMixturePrior(means={self.means.tolist()}, stds={self.stds.tolist()})"


class JointPrior:
    """Product of independent one-dimensional priors."""

    def __init__(self, priors: Sequence[Prior]) -> None:
        self.priors = list(priors)
        self.dim = len(self.priors)
        self.lower = np.array([p.lower for p in self.priors], dtype=float)
        self.upper = np.array([p.upper for p in self.priors], dtype=float)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        total = np.zeros(x.shape[0])
        for j, prior in enumerate(self.priors):
            total += prior.logpdf(x[:, j])
        return total

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.column_stack([prior.sample(n, rng) for prior in self.priors])

    def reflect(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fold points back into the support by mirror reflection at the bounds.

        Returns
        -------
        reflected : ndarray, shape (n, d)
            The points, inside the support.
        flipped : ndarray of bool, shape (n, d)
            ``True`` where a coordinate was reflected an odd number of times.

        Notes
        -----
        Reflection is equivalent to sampling the mirror-symmetric extension of the
        target, so no proposal is wasted outside the support.  A random-walk
        proposal with a *correlated* covariance is however not symmetric under
        the reflection of individual coordinates; the sampler uses ``flipped`` to
        apply the corresponding Hastings correction.
        """
        x = np.array(x, dtype=float, copy=True)
        flipped = np.zeros(x.shape, dtype=bool)
        lo, hi = self.lower, self.upper
        both = np.isfinite(lo) & np.isfinite(hi)
        if both.any():
            width = hi[both] - lo[both]
            t = np.mod(x[:, both] - lo[both], 2.0 * width)
            odd = t > width
            x[:, both] = lo[both] + np.where(odd, 2.0 * width - t, t)
            flipped[:, both] = odd
        only_lo = np.isfinite(lo) & ~np.isfinite(hi)
        if only_lo.any():
            flipped[:, only_lo] = x[:, only_lo] < lo[only_lo]
            x[:, only_lo] = lo[only_lo] + np.abs(x[:, only_lo] - lo[only_lo])
        only_hi = ~np.isfinite(lo) & np.isfinite(hi)
        if only_hi.any():
            flipped[:, only_hi] = x[:, only_hi] > hi[only_hi]
            x[:, only_hi] = hi[only_hi] - np.abs(hi[only_hi] - x[:, only_hi])
        return x, flipped
