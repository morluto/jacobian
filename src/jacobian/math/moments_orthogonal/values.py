"""Provider-independent native values for moments and orthogonal polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_MOMENTS = 64
MAX_POLYNOMIAL_COUNT = 32
MAX_HANKEL_DIMENSION = 32
MAX_RECURRENCE_ORDER = 16
MAX_QUADRATURE_POINTS = 16


@dataclass(frozen=True, slots=True)
class HankelMatrix:
    """The exact Hankel matrix built from a moment sequence."""

    matrix: tuple[tuple[Fraction, ...], ...]
    moments: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class RecurrenceCoefficients:
    """Three-term recurrence coefficients for a monic orthogonal family."""

    alpha: tuple[Fraction, ...]
    beta: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class JacobiMatrix:
    """The symmetric tridiagonal Jacobi matrix from recurrence coefficients."""

    diagonal: tuple[Fraction, ...]
    off_diagonal: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class ChristoffelDarbouxKernel:
    """The Christoffel-Darboux kernel summed over orthogonal polynomials."""

    kernel: Fraction
    polynomials_evaluated: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class GaussianQuadrature:
    """Gaussian quadrature nodes and weights from the Golub-Welsch algorithm.

    The Golub-Welsch eigenvalue decomposition runs in IEEE doubles; each
    returned value is the exact dyadic rational image of one computed double,
    so results stay canonical and reconstructible without JSON floats.
    """
    nodes: tuple[Fraction, ...]
    weights: tuple[Fraction, ...]


class RecurrenceCoefficientsValue(StrictModel):
    """Canonical exact three-term recurrence coefficients (alpha, beta).

    The domain-owned wire value for one recurrence pair: producers return
    it and Jacobi, Christoffel-Darboux, and quadrature consumers accept it
    unchanged.
    """

    alpha: tuple[CanonicalRational, ...]
    beta: tuple[CanonicalRational, ...]


__all__ = [
    "MAX_HANKEL_DIMENSION",
    "MAX_MOMENTS",
    "MAX_POLYNOMIAL_COUNT",
    "MAX_QUADRATURE_POINTS",
    "MAX_RECURRENCE_ORDER",
    "ChristoffelDarbouxKernel",
    "GaussianQuadrature",
    "HankelMatrix",
    "JacobiMatrix",
    "RecurrenceCoefficients",
    "RecurrenceCoefficientsValue",
]
