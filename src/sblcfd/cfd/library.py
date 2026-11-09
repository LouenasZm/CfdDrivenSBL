r"""Symbolic dictionary of corrective terms.

The corrections considered here have the form used for the intermittency
transport equation,

.. math::

    R = \Big(\sum_i \Gamma_i\, p_i(\lambda)\Big)\,
        \exp\Big(-\sum_j \Gamma_j\, \phi_j(\eta)\Big),

a polynomial prefactor that sets the amplitude and sign of the correction times an
exponential that sets its spatial support.  The library keeps the bookkeeping in
one place: the names and bounds of the coefficients, the evaluation of a
correction from flow features, and the printing of an identified model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..nsbl import Parameter

__all__ = ["Correction", "CorrectionLibrary", "Monomial", "monomial", "transition_library"]


@dataclass(frozen=True)
class Monomial:
    """Product of integer powers of named features, e.g. ``eta1^2 * eta2``."""

    powers: tuple[tuple[str, int], ...] = ()

    def evaluate(self, features: Mapping[str, np.ndarray]) -> np.ndarray | float:
        value: np.ndarray | float = 1.0
        for name, power in self.powers:
            value = value * features[name] ** power
        return value

    def label(self, symbols: Mapping[str, str] | None = None, latex: bool = False) -> str:
        if not self.powers:
            return "1"
        parts = []
        for name, power in self.powers:
            symbol = (symbols or {}).get(name, name)
            if power == 1:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}^{{{power}}}" if latex else f"{symbol}^{power}")
        return (" " if latex else "*").join(parts)


def monomial(**powers: int) -> Monomial:
    """``monomial(eta1=2, eta2=1)`` is the monomial ``eta1^2 * eta2``."""
    return Monomial(tuple((name, int(power)) for name, power in powers.items() if power != 0))


@dataclass
class Correction:
    """One corrective term: polynomial prefactor times a decaying exponential.

    Attributes
    ----------
    name : str
    prefactor : list of Monomial
        Terms of the amplitude polynomial.
    exponent : list of Monomial
        Terms of the argument of the exponential.
    """

    name: str
    prefactor: list[Monomial] = field(default_factory=list)
    exponent: list[Monomial] = field(default_factory=list)

    @property
    def n_parameters(self) -> int:
        return len(self.prefactor) + len(self.exponent)

    def evaluate(self, coefficients: np.ndarray, features: Mapping[str, np.ndarray]) -> np.ndarray:
        """Value of the correction.

        ``coefficients`` has shape ``(..., n_parameters)``, prefactor terms first;
        the features must broadcast against its leading dimensions.
        """
        coefficients = np.asarray(coefficients, dtype=float)
        n_pre = len(self.prefactor)
        amplitude = sum(
            coefficients[..., i] * m.evaluate(features) for i, m in enumerate(self.prefactor)
        )
        decay = sum(
            coefficients[..., n_pre + j] * m.evaluate(features) for j, m in enumerate(self.exponent)
        )
        return amplitude * np.exp(-decay)


class CorrectionLibrary:
    """Ordered collection of corrections sharing one coefficient vector.

    Parameters
    ----------
    corrections : sequence of Correction
    symbols : mapping, optional
        Display names of the features, e.g. ``{"eta1": r"\\eta_1"}`` for LaTeX output.
    """

    def __init__(
        self, corrections: Sequence[Correction], symbols: Mapping[str, str] | None = None
    ) -> None:
        self.corrections = list(corrections)
        self.symbols = dict(symbols or {})
        self.slices: dict[str, slice] = {}
        self.names: list[str] = []
        self.kinds: list[str] = []  # "constant", "prefactor" or "exponent"
        start = 0
        for c in self.corrections:
            self.slices[c.name] = slice(start, start + c.n_parameters)
            start += c.n_parameters
            for m in c.prefactor:
                self.names.append(f"{c.name}[{m.label()}]")
                self.kinds.append("constant" if not m.powers else "prefactor")
            for m in c.exponent:
                self.names.append(f"{c.name}[exp:{m.label()}]")
                self.kinds.append("exponent")
        self.n_parameters = start

    def parameters(
        self,
        *,
        constant: tuple[float, float] = (0.0, 1.0),
        prefactor: tuple[float, float] = (-1.0, 1.0),
        exponent: tuple[float, float] = (0.0, 1.0),
    ) -> list[Parameter]:
        """Questionable parameters with the bounds of each kind of coefficient.

        Exponent coefficients must be non-negative for the exponential to stay in
        ``(0, 1]``; they are therefore sign-constrained and receive half-normal ARD
        priors.
        """
        bounds = {"constant": constant, "prefactor": prefactor, "exponent": exponent}
        return [Parameter(n, *bounds[k]) for n, k in zip(self.names, self.kinds)]

    def evaluate(
        self, coefficients: np.ndarray, features: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Evaluate every correction; returns ``{name: values}``."""
        coefficients = np.asarray(coefficients, dtype=float)
        return {
            c.name: c.evaluate(coefficients[..., self.slices[c.name]], features)
            for c in self.corrections
        }

    def format(
        self,
        coefficients: np.ndarray,
        *,
        active: np.ndarray | None = None,
        std: np.ndarray | None = None,
        latex: bool = False,
        precision: int = 3,
    ) -> str:
        """Closed form of the model, pruned terms omitted.

        Parameters
        ----------
        coefficients : array_like, shape (n_parameters,)
        active : array_like of bool, optional
            Terms to print; defaults to the non-zero coefficients.
        std : array_like, optional
            Posterior standard deviations, printed as ``value(+-std)``.
        latex : bool
        precision : int
            Significant digits.
        """
        values = np.asarray(coefficients, dtype=float)
        active = values != 0.0 if active is None else np.asarray(active, dtype=bool)

        def number(i: int, magnitude: bool) -> str:
            v = abs(values[i]) if magnitude else values[i]
            text = f"{v:.{precision}g}"
            if std is not None:
                text += f"(±{std[i]:.{max(precision - 1, 1)}g})"
            return text

        def term(i: int, m: Monomial, first: bool) -> str:
            sign = "-" if values[i] < 0 else "+"
            lead = ("-" if values[i] < 0 else "") if first else f" {sign} "
            label = m.label(self.symbols, latex)
            body = number(i, True) if label == "1" else f"{number(i, True)} {label}"
            return lead + body

        lines = []
        for c in self.corrections:
            offset = self.slices[c.name].start
            pre = [(offset + i, m) for i, m in enumerate(c.prefactor) if active[offset + i]]
            n_pre = len(c.prefactor)
            exp = [
                (offset + n_pre + j, m)
                for j, m in enumerate(c.exponent)
                if active[offset + n_pre + j]
            ]
            if not pre:
                lines.append(f"{c.name} = 0")
                continue
            amplitude = "".join(term(i, m, k == 0) for k, (i, m) in enumerate(pre))
            text = f"{c.name} = ({amplitude})" if len(pre) > 1 else f"{c.name} = {amplitude}"
            if exp:
                decay = "".join(term(i, m, k == 0) for k, (i, m) in enumerate(exp))
                times = r" \, " if latex else " * "
                text += f"{times}exp(-[{decay}])"
            lines.append(text)
        return "\n".join(lines)


def transition_library() -> CorrectionLibrary:
    r"""Dictionary of the augmented intermittency equation (21 coefficients).

    * production correction :math:`R^P`: cubic polynomial of the acceleration
      parameter :math:`\lambda_\theta` times
      :math:`\exp(-\sum\Gamma_i\phi_i(\eta_1,\eta_2))`, seven monomials;
    * destruction correction :math:`R^E`: cubic polynomial of
      :math:`\lambda_\theta` times :math:`\exp(-\sum\Gamma_j\psi_j(\eta_1,\eta_3))`,
      six monomials,

    with :math:`\eta_1 = \mu_t/\mu`, :math:`\eta_2 = d_w/(d_w+\sqrt{k}/\omega)` and
    :math:`\eta_3 = \rho d_w^2\Omega/\mu`.
    """
    polynomial = [monomial(), monomial(lam=1), monomial(lam=2), monomial(lam=3)]
    production = Correction(
        "RP",
        polynomial,
        [
            monomial(eta1=1),
            monomial(eta2=1),
            monomial(eta1=2),
            monomial(eta1=1, eta2=1),
            monomial(eta2=2),
            monomial(eta1=2, eta2=1),
            monomial(eta1=1, eta2=2),
        ],
    )
    destruction = Correction(
        "RE",
        polynomial,
        [
            monomial(eta1=1, eta3=1),
            monomial(eta1=1, eta3=2),
            monomial(eta1=2, eta3=1),
            monomial(eta1=2, eta3=2),
            monomial(eta1=3, eta3=1),
            monomial(eta1=1, eta3=3),
        ],
    )
    return CorrectionLibrary([production, destruction])
