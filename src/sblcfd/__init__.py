"""Nonlinear sparse Bayesian learning (NSBL) for CFD-driven model discovery.

The core implements the semi-analytical framework of Sandhu, Khalil, Pettit,
Poirel and Sarkar, "Nonlinear sparse Bayesian learning for physics-based models",
J. Comput. Phys. 426 (2021) 109728.  :mod:`sblcfd.cfd` adds what a surrogate-driven
RANS application needs on top: batched POD-Kriging emulators and field likelihoods.
"""

from .evidence import ARDEvidence, GammaHyperprior
from .mixture import GaussianMixture, fit_mixture
from .nsbl import NSBL, Calibration, NSBLResult, Parameter, model_average, questionable
from .optimize import Optimum, maximize_evidence
from .priors import GaussianMixturePrior, HalfNormal, JointPrior, Normal, Prior, Uniform
from .smc import SMCResult, SMCSampler

__version__ = "0.1.0"

__all__ = [
    "NSBL",
    "ARDEvidence",
    "Calibration",
    "GammaHyperprior",
    "GaussianMixture",
    "GaussianMixturePrior",
    "HalfNormal",
    "JointPrior",
    "NSBLResult",
    "Normal",
    "Optimum",
    "Parameter",
    "Prior",
    "SMCResult",
    "SMCSampler",
    "Uniform",
    "fit_mixture",
    "maximize_evidence",
    "model_average",
    "questionable",
]
