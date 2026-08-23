"""Provider-independent native values for moments and orthogonal polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Self

from pydantic import model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_MOMENTS = 64
MAX_POLYNOMIAL_COUNT = 32
MAX_HANKEL_DIMENSION = 32
MAX_RECURRENCE_ORDER = 16
MAX_QUADRATURE_POINTS = 16
MAX_RATIONAL_DIGITS = 4_096


@dataclass(frozen=True, slots=True)
class HankelMatrix:
    """The exact Hankel matrix built from a moment sequence."""

    matrix: tuple[tuple[Fraction, ...], ...]
    moments: tuple[Fraction, ...]


class RecurrenceCoefficients(StrictModel):
    """Canonical three-term recurrence coefficients of a monic orthogonal family.

    ``alpha`` carries the shift coefficients and ``beta`` the squared-norm
    ratios with positive ``beta_0``; the recurrence producer returns this
    value and the Jacobi-matrix, Christoffel-Darboux, and Gaussian-quadrature
    consumers accept it unchanged.
    """

    alpha: tuple[CanonicalRational, ...]
    beta: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_positive_definite(self) -> Self:
        if not 1 <= len(self.beta) <= MAX_RECURRENCE_ORDER:
            raise ValueError("beta must contain between 1 and 16 entries")
        if not 0 <= len(self.alpha) <= MAX_RECURRENCE_ORDER:
            raise ValueError("alpha out of range")
        if len(self.alpha) != len(self.beta) and len(self.alpha) != len(self.beta) - 1:
            raise ValueError("alpha must have length len(beta)-1 or len(beta)")
        if self.beta[0].as_fraction() <= 0:
            raise ValueError(
                "beta_0 (the zeroth moment of a positive functional) must be positive"
            )
        # beta_1, ..., are squared-norm ratios of a positive-definite family;
        # every entry after beta_0 must be positive, including the trailing
        # entry of a partial recurrence with len(alpha) == len(beta) - 1.
        for index in range(1, len(self.beta)):
            if self.beta[index].as_fraction() <= 0:
                raise ValueError(
                    "subdiagonal beta entries must be positive squared-norm ratios"
                )
        for value in (*self.alpha, *self.beta):
            require_bounded_rational(
                value, max_digits=MAX_RATIONAL_DIGITS, label="coefficient"
            )
        return self


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

    The nodes and weights are *approximate* floating-point approximations of
    the exact Gaussian quadrature rule: the exact nodes are algebraic numbers
    (roots of the orthogonal polynomial) and generally irrational.  The result
    carries an explicit approximation contract via ``is_approximate`` and
    ``precision`` and must not be treated as exact rational nodes.
    """

    nodes: tuple[Fraction, ...]
    weights: tuple[Fraction, ...]
    is_approximate: bool = True
    precision: str = "FLOAT64"


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
]
