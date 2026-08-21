"""Typed wire contracts for moment-functional operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.moments_orthogonal.values import (
    MAX_HANKEL_ORDER,
    MAX_MOMENT_DEGREE,
    MAX_POLYNOMIAL_DEGREE,
    ChristoffelDarbouxKernel,
    GaussianQuadratureRule,
    HankelMomentMatrix,
    OrthogonalPolynomialFamily,
    ThreeTermRecurrence,
)


class HankelRequest(StrictModel):
    """Compute the Hankel matrix H_r from a moment prefix."""

    moments: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=MAX_MOMENT_DEGREE + 1)
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER)
    variable: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order + 1
        if len(self.moments) < needed:
            raise ValueError(
                f"need at least {needed} moments for order {self.order}, got {len(self.moments)}"
            )
        return self


class ShiftedHankelRequest(StrictModel):
    """Compute the shifted Hankel matrix H_r^(1)[i,j] = mu_(i+j+1)."""

    moments: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=MAX_MOMENT_DEGREE + 1)
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER)
    variable: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order + 2
        if len(self.moments) < needed:
            raise ValueError(
                f"need at least {needed} moments for shifted order {self.order}, got {len(self.moments)}"
            )
        return self


class OrthogonalPolynomialRequest(StrictModel):
    """Compute monic orthogonal polynomials from moments."""

    moments: tuple[CanonicalRational, ...] = Field(min_length=2, max_length=MAX_MOMENT_DEGREE + 1)
    max_degree: int = Field(ge=0, le=MAX_POLYNOMIAL_DEGREE)
    variable: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.max_degree + 1
        if len(self.moments) < needed:
            raise ValueError(
                f"need at least {needed} moments for degree {self.max_degree}, got {len(self.moments)}"
            )
        return self


class RecurrenceRequest(StrictModel):
    """Compute three-term recurrence coefficients from a family."""

    family: OrthogonalPolynomialFamily


class ChristoffelDarbouxRequest(StrictModel):
    """Compute the Christoffel-Darboux kernel."""

    family: OrthogonalPolynomialFamily
    degree: int = Field(ge=0)


class JacobiMatrixRequest(StrictModel):
    """Compute the finite Jacobi matrix."""

    family: OrthogonalPolynomialFamily


class GaussianQuadratureRequest(StrictModel):
    """Compute an exact Gaussian quadrature rule."""

    moments: tuple[CanonicalRational, ...] = Field(min_length=2, max_length=MAX_MOMENT_DEGREE + 1)
    order: int = Field(ge=1, le=16)
    variable: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order
        if len(self.moments) < needed:
            raise ValueError(
                f"need at least {needed} moments for quadrature order {self.order}, got {len(self.moments)}"
            )
        return self


__all__ = [
    "ChristoffelDarbouxRequest",
    "GaussianQuadratureRequest",
    "HankelRequest",
    "JacobiMatrixRequest",
    "OrthogonalPolynomialRequest",
    "RecurrenceRequest",
    "ShiftedHankelRequest",
]
