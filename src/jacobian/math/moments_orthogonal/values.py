"""Provider-independent native values for moments and orthogonal polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Self

from pydantic import model_validator

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


class RecurrenceCoefficients(StrictModel):
    """Three-term recurrence coefficients for a monic orthogonal family.

    The one domain-owned value: produced by the exact Gram-Schmidt kernel,
    carried by the wire result, and accepted unchanged by the Jacobi-matrix,
    Christoffel-Darboux, and quadrature consumers on both the native and MCP
    surfaces.
    """

    alpha: tuple[CanonicalRational, ...]
    beta: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_valid(self) -> Self:
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
        # Every beta after beta_0 is a squared-norm ratio of a positive-
        # definite sequence (the trailing entry of an odd-length prefix
        # included), so each must be positive.
        for entry in self.beta[1:]:
            if entry.as_fraction() <= 0:
                raise ValueError(
                    "subdiagonal beta entries must be positive squared-norm ratios"
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
    """

    nodes: tuple[Fraction, ...]
    weights: tuple[Fraction, ...]


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
