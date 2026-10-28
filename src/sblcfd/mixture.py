"""Gaussian mixtures with shared covariances.

The whole NSBL machinery rests on one object: an *unnormalised* Gaussian mixture

.. math::

    g(\\phi) = \\sum_{k=1}^{K} a_k \\, \\mathcal{N}(\\phi \\mid \\mu_k, \\Sigma_{c(k)})

that approximates the product of the likelihood and the known prior (Eq. 3 of
Sandhu et al., 2021).  Kernels may share covariance matrices: a kernel density
estimate has ``K`` kernels and a single covariance, an EM fit has one covariance
per kernel.  Exploiting that sharing is what keeps the evidence, its gradient and
its Hessian at ``O(G n^3 + K n^2)`` per evaluation, with ``G`` the number of
distinct covariances.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence

import numpy as np
from scipy.linalg import LinAlgError, cho_solve, cholesky, solve_triangular
from scipy.special import logsumexp

LOG_2PI = float(np.log(2.0 * np.pi))

__all__ = ["GaussianMixture", "fit_mixture"]


def _cholesky(cov: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor, adding the smallest jitter that makes ``cov`` positive definite."""
    cov = 0.5 * (cov + cov.T)
    try:
        return cholesky(cov, lower=True, check_finite=False)
    except LinAlgError:
        pass
    scale = max(float(np.trace(cov)) / cov.shape[0], np.finfo(float).tiny)
    eye = np.eye(cov.shape[0])
    for exponent in range(-14, -2):
        try:
            return cholesky(cov + (10.0**exponent) * scale * eye, lower=True, check_finite=False)
        except LinAlgError:
            continue
    raise LinAlgError("covariance matrix is not positive definite, even after regularisation")


def _log_gauss_from_chol(chol: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """``log N(diff_k | 0, L L^T)`` for every row of ``diff``."""
    z = solve_triangular(chol, diff.T, lower=True, check_finite=False)
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    return -0.5 * (diff.shape[1] * LOG_2PI + logdet + np.einsum("ij,ij->j", z, z))


class GaussianMixture:
    r"""Unnormalised Gaussian mixture :math:`\sum_k a_k \mathcal{N}(\mu_k, \Sigma_{c(k)})`.

    Parameters
    ----------
    means : array_like, shape (K, d)
        Kernel centres.
    covariances : array_like, shape (d, d) or (G, d, d)
        The ``G`` distinct covariance matrices.
    log_weights : array_like, shape (K,), optional
        Logarithm of the amplitudes :math:`a_k`.  They need not sum to one: their
        sum is the total mass of the mixture, which carries the absolute scale of
        the model evidence.  Defaults to equal weights with unit mass.
    cov_index : array_like of int, shape (K,), optional
        Index of the covariance used by each kernel.  May be omitted when
        ``G == 1`` (all kernels share it) or ``G == K`` (one each).

    Notes
    -----
    Kernels are stored sorted by covariance index, so that kernels sharing a
    covariance are contiguous.  The order of the kernels may therefore differ from
    the order of the inputs.
    """

    def __init__(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        log_weights: np.ndarray | None = None,
        cov_index: np.ndarray | None = None,
    ) -> None:
        means = np.atleast_2d(np.asarray(means, dtype=float))
        n_kernels, dim = means.shape
        cov = np.asarray(covariances, dtype=float)
        if cov.ndim == 2:
            cov = cov[None]
        if cov.ndim != 3 or cov.shape[1:] != (dim, dim):
            raise ValueError(f"covariances must have shape (G, {dim}, {dim}), got {cov.shape}")
        n_cov = cov.shape[0]

        if cov_index is None:
            if n_cov == 1:
                cov_index = np.zeros(n_kernels, dtype=int)
            elif n_cov == n_kernels:
                cov_index = np.arange(n_kernels)
            else:
                raise ValueError("cov_index is required when 1 < G < K")
        cov_index = np.asarray(cov_index, dtype=int)
        if cov_index.shape != (n_kernels,):
            raise ValueError("cov_index must have one entry per kernel")
        if cov_index.min() < 0 or cov_index.max() >= n_cov:
            raise ValueError("cov_index refers to a covariance that does not exist")

        if log_weights is None:
            log_weights = np.full(n_kernels, -np.log(n_kernels))
        log_weights = np.asarray(log_weights, dtype=float)
        if log_weights.shape != (n_kernels,):
            raise ValueError("log_weights must have one entry per kernel")

        if np.any(np.diff(cov_index) < 0):
            order = np.argsort(cov_index, kind="stable")
            means, log_weights, cov_index = means[order], log_weights[order], cov_index[order]

        self.means = means
        self.covariances = 0.5 * (cov + np.swapaxes(cov, 1, 2))
        self.log_weights = log_weights
        self.cov_index = cov_index
        counts = np.bincount(cov_index, minlength=n_cov)
        self._starts = np.concatenate([[0], np.cumsum(counts)])

    # ------------------------------------------------------------------ #
    #  basic properties
    # ------------------------------------------------------------------ #
    @property
    def n_components(self) -> int:
        """Number of kernels ``K``."""
        return self.means.shape[0]

    @property
    def dim(self) -> int:
        """Dimension ``d`` of the space the mixture lives in."""
        return self.means.shape[1]

    @property
    def n_covariances(self) -> int:
        """Number of distinct covariance matrices ``G``."""
        return self.covariances.shape[0]

    @property
    def is_tied(self) -> bool:
        """``True`` when every kernel shares one covariance (kernel density estimate)."""
        return self.n_covariances == 1

    @property
    def log_mass(self) -> float:
        """Logarithm of the total mass :math:`\\log \\sum_k a_k`."""
        return float(logsumexp(self.log_weights))

    @property
    def weights(self) -> np.ndarray:
        """Normalised weights :math:`a_k / \\sum_j a_j`."""
        return np.exp(self.log_weights - self.log_mass)

    def groups(self) -> Iterator[tuple[int, slice]]:
        """Yield ``(g, slice)`` pairs: the kernels in ``slice`` use covariance ``g``."""
        for g in range(self.n_covariances):
            if self._starts[g + 1] > self._starts[g]:
                yield g, slice(int(self._starts[g]), int(self._starts[g + 1]))

    def normalized(self) -> GaussianMixture:
        """Same mixture rescaled to unit mass (a probability density)."""
        return GaussianMixture(
            self.means, self.covariances, self.log_weights - self.log_mass, self.cov_index
        )

    def __repr__(self) -> str:
        return (
            f"GaussianMixture(K={self.n_components}, d={self.dim}, "
            f"G={self.n_covariances}, log_mass={self.log_mass:.4g})"
        )

    # ------------------------------------------------------------------ #
    #  moments, density, sampling
    # ------------------------------------------------------------------ #
    def mean(self) -> np.ndarray:
        """Mean of the normalised mixture."""
        return self.weights @ self.means

    def covariance(self) -> np.ndarray:
        """Covariance of the normalised mixture (within- plus between-kernel parts)."""
        w = self.weights
        centred = self.means - w @ self.means
        cov = (centred * w[:, None]).T @ centred
        for g, sl in self.groups():
            cov += w[sl].sum() * self.covariances[g]
        return cov

    def std(self) -> np.ndarray:
        """Marginal standard deviations of the normalised mixture."""
        return np.sqrt(np.diag(self.covariance()))

    def logpdf(
        self, x: np.ndarray, *, normalized: bool = False, chunk: int = 4_000_000
    ) -> np.ndarray:
        """Log-density at the rows of ``x``.

        Parameters
        ----------
        x : array_like, shape (n, d)
        normalized : bool
            If ``True`` the mixture is rescaled to unit mass first.
        chunk : int
            Upper bound on the size of the intermediate ``points x kernels`` block.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        out = np.full(x.shape[0], -np.inf)
        for g, sl in self.groups():
            chol = _cholesky(self.covariances[g])
            logdet = 2.0 * np.log(np.diag(chol)).sum()
            xw = solve_triangular(chol, x.T, lower=True, check_finite=False).T
            mw = solve_triangular(chol, self.means[sl].T, lower=True, check_finite=False).T
            m_sq = np.einsum("ij,ij->i", mw, mw)
            logw = self.log_weights[sl]
            step = max(1, chunk // max(mw.shape[0], 1))
            for start in range(0, x.shape[0], step):
                block = xw[start : start + step]
                d2 = np.einsum("ij,ij->i", block, block)[:, None] + m_sq[None, :]
                d2 -= 2.0 * block @ mw.T
                np.maximum(d2, 0.0, out=d2)
                lp = logsumexp(logw[None, :] - 0.5 * d2, axis=1)
                lp -= 0.5 * (self.dim * LOG_2PI + logdet)
                out[start : start + step] = np.logaddexp(out[start : start + step], lp)
        if normalized:
            out -= self.log_mass
        return out

    def pdf(self, x: np.ndarray, *, normalized: bool = False) -> np.ndarray:
        """Density at the rows of ``x`` (see :meth:`logpdf`)."""
        return np.exp(self.logpdf(x, normalized=normalized))

    def marginal_pdf(self, index: int, grid: np.ndarray) -> np.ndarray:
        """Normalised one-dimensional marginal density of coordinate ``index`` on ``grid``."""
        grid = np.asarray(grid, dtype=float)
        w = self.weights
        out = np.zeros_like(grid)
        for g, sl in self.groups():
            sd = np.sqrt(self.covariances[g][index, index])
            z = (grid[:, None] - self.means[sl, index][None, :]) / sd
            out += np.exp(-0.5 * z * z) @ w[sl] / (sd * np.sqrt(2.0 * np.pi))
        return out

    def sample(self, n: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """Draw ``n`` points from the normalised mixture."""
        rng = np.random.default_rng(rng)
        component = rng.choice(self.n_components, size=n, p=self.weights)
        out = self.means[component].copy()
        noise = rng.standard_normal((n, self.dim))
        group = self.cov_index[component]
        for g, _ in self.groups():
            mask = group == g
            if mask.any():
                out[mask] += noise[mask] @ _cholesky(self.covariances[g]).T
        return out

    # ------------------------------------------------------------------ #
    #  closed-form operations
    # ------------------------------------------------------------------ #
    def marginal(self, index: Sequence[int]) -> GaussianMixture:
        """Mixture of the coordinates in ``index``, the others being integrated out."""
        index = np.asarray(index, dtype=int)
        cov = self.covariances[:, index[:, None], index[None, :]]
        return GaussianMixture(self.means[:, index], cov, self.log_weights, self.cov_index)

    def condition(self, index: Sequence[int], values: np.ndarray | float = 0.0) -> GaussianMixture:
        r"""Slice of the mixture at ``x[index] = values``.

        The result lives on the remaining coordinates and is *not* renormalised:
        its mass is the marginal density of ``x[index]`` evaluated at ``values``.
        Conditioning on a coefficient being exactly zero is how a pruned model is
        obtained without drawing new samples.
        """
        index = np.asarray(index, dtype=int)
        rest = np.setdiff1d(np.arange(self.dim), index)
        values = np.broadcast_to(np.asarray(values, dtype=float), index.shape)

        means = np.empty((self.n_components, rest.size))
        covs = np.empty((self.n_covariances, rest.size, rest.size))
        logw = self.log_weights.copy()
        for g, sl in self.groups():
            cov = self.covariances[g]
            chol = _cholesky(cov[np.ix_(index, index)])
            gain = cho_solve((chol, True), cov[np.ix_(index, rest)], check_finite=False)
            covs[g] = cov[np.ix_(rest, rest)] - cov[np.ix_(rest, index)] @ gain
            diff = values[None, :] - self.means[sl][:, index]
            logw[sl] += _log_gauss_from_chol(chol, diff)
            means[sl] = self.means[sl][:, rest] + diff @ gain
        for g in range(self.n_covariances):
            if self._starts[g + 1] == self._starts[g]:
                covs[g] = np.eye(rest.size)
        return GaussianMixture(means, covs, logw, self.cov_index)

    def multiply_gaussian(
        self, index: Sequence[int], mean: np.ndarray | float, cov: np.ndarray
    ) -> GaussianMixture:
        r"""Product with a Gaussian factor :math:`\mathcal{N}(x_{index} \mid c, C)`.

        Every kernel is updated in closed form (a Kalman update with observation
        operator ``x -> x[index]``):

        .. math::

            a_k \mathcal{N}(x \mid \mu_k, \Sigma)\,\mathcal{N}(x_S \mid c, C)
            = a_k \mathcal{N}(\mu_{k,S} \mid c, B)\; \mathcal{N}(x \mid m_k, P),
            \qquad B = \Sigma_{SS} + C .

        The mass of the returned mixture is the integral of the product.  With
        ``c = 0`` and ``C = A^{-1}`` this is the NSBL evidence (Eq. 9) and the
        normalised result is the NSBL posterior (Eqs. 11-12).

        Parameters
        ----------
        index : sequence of int
            Coordinates the factor acts on.
        mean : float or array_like, shape (len(index),)
        cov : array_like
            Either the full covariance ``C`` or, if one-dimensional, its diagonal.
        """
        index = np.asarray(index, dtype=int)
        centre = np.broadcast_to(np.asarray(mean, dtype=float), index.shape)
        cov = np.asarray(cov, dtype=float)
        factor_cov = np.diag(cov) if cov.ndim == 1 else cov

        means = np.empty_like(self.means)
        covs = self.covariances.copy()
        logw = self.log_weights.copy()
        for g, sl in self.groups():
            sigma = self.covariances[g]
            chol = _cholesky(sigma[np.ix_(index, index)] + factor_cov)
            gain = cho_solve((chol, True), sigma[index, :], check_finite=False)  # B^-1 Sigma_S.
            post = sigma - sigma[:, index] @ gain
            covs[g] = 0.5 * (post + post.T)
            diff = centre[None, :] - self.means[sl][:, index]
            logw[sl] += _log_gauss_from_chol(chol, diff)
            means[sl] = self.means[sl] + diff @ gain
        return GaussianMixture(means, covs, logw, self.cov_index)

    # ------------------------------------------------------------------ #
    #  constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_gaussian(
        cls, mean: np.ndarray, cov: np.ndarray, log_mass: float = 0.0
    ) -> GaussianMixture:
        """Single Gaussian (``K = 1``): a Laplace / moment-matched approximation."""
        mean = np.atleast_1d(np.asarray(mean, dtype=float))
        return cls(mean[None, :], np.asarray(cov, dtype=float), np.array([log_mass]))

    @classmethod
    def from_moments(
        cls,
        samples: np.ndarray,
        *,
        symmetric_dims: Sequence[int] | None = None,
        log_mass: float = 0.0,
    ) -> GaussianMixture:
        """Single Gaussian matched to the first two moments of the samples.

        Along ``symmetric_dims`` the moments are those of the samples mirrored about
        zero: zero mean, second moment about zero, no correlation with the other
        coordinates.  See :meth:`from_em` for when to mirror.
        """
        x = np.atleast_2d(np.asarray(samples, dtype=float))
        mean = x.mean(axis=0)
        cov = np.atleast_2d(np.cov(x, rowvar=False))
        if symmetric_dims is not None and len(symmetric_dims) > 0:
            dims = np.asarray(symmetric_dims, dtype=int)
            mean[dims] = 0.0
            cov[dims, :] = 0.0
            cov[:, dims] = 0.0
            cov[dims, dims] = (x[:, dims] ** 2).mean(axis=0)
        return cls.from_gaussian(mean, cov, log_mass)

    @classmethod
    def from_kde(
        cls,
        samples: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        bandwidth: str | float = "scott",
        factor: float = 1.0,
        shrinkage: bool = False,
        log_mass: float = 0.0,
    ) -> GaussianMixture:
        r"""Kernel density estimate: one Gaussian kernel per sample, one shared covariance.

        The kernel covariance is :math:`h^2 S`, with :math:`S` the sample covariance
        and :math:`h` the bandwidth factor.  This is the construction used in the
        NSBL paper (``scipy.stats.gaussian_kde`` with Scott's rule).

        Parameters
        ----------
        samples : array_like, shape (N, d)
            Stationary draws from ``likelihood x known prior``.
        weights : array_like, shape (N,), optional
            Importance weights of the draws.
        bandwidth : {"scott", "silverman"} or float
            Rule for the factor :math:`h`, or its value.
        factor : float
            Multiplier applied to :math:`h`.  The evidence is an integral of the
            density, so it tolerates a smaller bandwidth than a pointwise density
            estimate does; values below one trade variance for a smaller bias.
        shrinkage : bool
            If ``True`` the centres are contracted towards the sample mean by
            :math:`\sqrt{1 - h^2}` so that the mixture reproduces the sample mean
            *and* covariance exactly (West, 1993).  Do not use it for
            sign-constrained coefficients whose samples pile up against zero: the
            contraction moves the kernels away from the boundary.
        log_mass : float
            Logarithm of the integral of ``likelihood x known prior``, if known.
        """
        x = np.atleast_2d(np.asarray(samples, dtype=float))
        n, dim = x.shape
        if n <= dim:
            raise ValueError(f"need more samples ({n}) than dimensions ({dim}) for a KDE")
        if weights is None:
            w = np.full(n, 1.0 / n)
        else:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
        n_eff = 1.0 / np.sum(w * w)

        if isinstance(bandwidth, str):
            if bandwidth == "scott":
                h = n_eff ** (-1.0 / (dim + 4))
            elif bandwidth == "silverman":
                h = (n_eff * (dim + 2) / 4.0) ** (-1.0 / (dim + 4))
            else:
                raise ValueError(f"unknown bandwidth rule '{bandwidth}'")
        else:
            h = float(bandwidth)
        h *= factor
        if h <= 0.0:
            raise ValueError("the bandwidth factor must be positive")

        centre = w @ x
        dev = x - centre
        sample_cov = (dev * w[:, None]).T @ dev / (1.0 - np.sum(w * w))
        if shrinkage:
            if h >= 1.0:
                raise ValueError("shrinkage requires a bandwidth factor below one")
            x = centre + np.sqrt(1.0 - h * h) * dev
        return cls(x, (h * h) * sample_cov, log_mass + np.log(w))

    @classmethod
    def from_em(
        cls,
        samples: np.ndarray,
        *,
        n_components: int | Sequence[int] = (1, 2, 3, 4, 6, 8),
        symmetric_dims: Sequence[int] | None = None,
        n_init: int = 2,
        reg_covar: float = 1e-6,
        log_mass: float = 0.0,
        seed: int | None = None,
    ) -> GaussianMixture:
        """Full-covariance mixture fitted by expectation-maximisation (scikit-learn).

        Beyond eight dimensions or so a kernel density estimate needs a bandwidth
        comparable to the spread of the samples, and the evidence of a sparse model
        ends up resting on a handful of kernels: its estimate becomes too noisy to
        rank structures.  A few full-covariance kernels give a smooth evidence
        instead; this is what the NSBL paper recommends for large parameter spaces.

        Parameters
        ----------
        samples : array_like, shape (N, d)
        n_components : int or sequence of int
            Number of kernels, or candidates among which the BIC selects.
        symmetric_dims : sequence of int, optional
            Coordinates along which the samples are mirrored about zero (random
            signs) before fitting.  Use it for sign-constrained coefficients whose
            samples accumulate at zero: the evidence integral is unchanged by the
            mirroring (see ``docs/theory.md``) while the mirrored density is smooth
            across zero and therefore easy to fit.
        n_init : int
            Restarts of the EM algorithm per candidate.
        reg_covar : float
            Ridge added to the covariances, relative to the sample variances.
        log_mass : float
            Logarithm of the integral of ``likelihood x known prior``, if known.
        seed : int, optional
        """
        try:
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.mixture import GaussianMixture as _SklearnGMM
        except ImportError as exc:  # pragma: no cover - exercised only without sklearn
            raise ImportError(
                "the EM mixture needs scikit-learn: pip install scikit-learn"
            ) from exc

        x = np.atleast_2d(np.asarray(samples, dtype=float)).copy()
        rng = np.random.default_rng(seed)
        if symmetric_dims is not None and len(symmetric_dims) > 0:
            dims = np.asarray(symmetric_dims, dtype=int)
            x[:, dims] *= rng.choice([-1.0, 1.0], size=(x.shape[0], dims.size))

        centre, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        z = (x - centre) / scale

        candidates = [n_components] if np.isscalar(n_components) else list(n_components)
        best, best_bic = None, np.inf
        for k in candidates:
            model = _SklearnGMM(
                n_components=int(k),
                covariance_type="full",
                n_init=n_init,
                reg_covar=reg_covar,
                max_iter=500,
                random_state=int(rng.integers(2**31 - 1)),
            )
            with warnings.catch_warnings():
                # a fit that stops at max_iter is still a usable density estimate
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(z)
            bic = model.bic(z)
            if bic < best_bic:
                best, best_bic = model, bic

        means = best.means_ * scale + centre
        covs = best.covariances_ * scale[None, :, None] * scale[None, None, :]
        return cls(means, covs, log_mass + np.log(best.weights_))


def fit_mixture(samples: np.ndarray, method: str = "kde", **kwargs) -> GaussianMixture:
    """Build the Gaussian-mixture approximation of a sampled density.

    Parameters
    ----------
    samples : array_like, shape (N, d)
        Draws from ``likelihood x known prior``.
    method : {"kde", "em", "gaussian"}
        ``"kde"`` -- one kernel per sample (:meth:`GaussianMixture.from_kde`);
        ``"em"`` -- a few full-covariance kernels (:meth:`GaussianMixture.from_em`);
        ``"gaussian"`` -- a single moment-matched Gaussian.
    **kwargs
        Forwarded to the selected constructor.
    """
    if method == "kde":
        return GaussianMixture.from_kde(samples, **kwargs)
    if method == "em":
        return GaussianMixture.from_em(samples, **kwargs)
    if method == "gaussian":
        return GaussianMixture.from_moments(samples, **kwargs)
    raise ValueError(f"unknown mixture method '{method}'")
