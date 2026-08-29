"""Typed wire contracts for moment-functional operations."""

from __future__ import annotations

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.analysis.orthogonal_polynomials.values import (
    MAX_HANKEL_ORDER,
    MAX_POLYNOMIAL_DEGREE,
    MAX_QUADRATURE_ORDER,
    MomentFunctionalPrefix,
    OrthogonalPolynomialFamily,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"moments_orthogonal.{reason}", message)


class HankelRequest(StrictModel):
    """Compute the Hankel matrix H_r from a moment prefix."""

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER)


class ShiftedHankelRequest(StrictModel):
    """Compute the shifted Hankel matrix H_r^(1)[i,j] = mu_(i+j+1)."""

    prefix: MomentFunctionalPrefix
    # A shifted matrix of order r consumes mu_1..mu_(2r+1); the canonical
    # prefix holds at most 129 moments, so r = 64 could never validate and
    # must not be advertised as supported.
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER - 1)


class OrthogonalPolynomialRequest(StrictModel):
    """Compute monic orthogonal polynomials from moments."""

    prefix: MomentFunctionalPrefix
    max_degree: int = Field(ge=0, le=MAX_POLYNOMIAL_DEGREE)


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

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=1, le=MAX_QUADRATURE_ORDER)


__all__ = [
    "ChristoffelDarbouxRequest",
    "GaussianQuadratureRequest",
    "HankelRequest",
    "JacobiMatrixRequest",
    "OrthogonalPolynomialRequest",
    "RecurrenceRequest",
    "ShiftedHankelRequest",
]


MomentFunctionalPrefix.model_rebuild()
HankelRequest.model_rebuild()
ShiftedHankelRequest.model_rebuild()
OrthogonalPolynomialRequest.model_rebuild()
ChristoffelDarbouxRequest.model_rebuild()
GaussianQuadratureRequest.model_rebuild()
