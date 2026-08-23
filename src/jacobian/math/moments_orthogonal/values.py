"""Provider-independent native values for moments and orthogonal polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from fractions import Fraction as _Fraction

MAX_MOMENTS = 64
MAX_POLYNOMIAL_COUNT = 32
MAX_HANKEL_DIMENSION = 32
MAX_RECURRENCE_ORDER = 16
MAX_QUADRATURE_POINTS = 16

# Golub-Welsch converts admitted rationals to IEEE doubles; every accepted
# coefficient must convert to a finite nonzero double and every subdiagonal
# entry must stay far from both overflow and underflow so its square root is
# exact enough.
MAX_QUADRATURE_MAGNITUDE = _Fraction(10) ** 300
MIN_QUADRATURE_SUBDIAGONAL = _Fraction(1, 10**300)


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


__all__ = [
    "MAX_HANKEL_DIMENSION",
    "MAX_MOMENTS",
    "MAX_POLYNOMIAL_COUNT",
    "MAX_QUADRATURE_MAGNITUDE",
    "MAX_QUADRATURE_POINTS",
    "MAX_RECURRENCE_ORDER",
    "MIN_QUADRATURE_SUBDIAGONAL",
    "ChristoffelDarbouxKernel",
    "GaussianQuadrature",
    "HankelMatrix",
    "JacobiMatrix",
    "RecurrenceCoefficients",
]
