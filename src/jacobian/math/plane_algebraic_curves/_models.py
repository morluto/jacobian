"""Typed wire contracts for plane algebraic curve operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_VARS = 3
HOMOGENIZING_COORDINATE = "z"
_MAX_TERMS = 256
_MAX_EXPONENT = 64
_MAX_COEFFICIENT_DIGITS = 128


def _require_curve_polynomial(polynomial: RationalPolynomial) -> None:
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_TERMS,
        maximum_exponent=_MAX_EXPONENT,
        maximum_coefficient_digits=_MAX_COEFFICIENT_DIGITS,
        label="curve polynomial",
    )
    if any(sum(term.exponents) > _MAX_EXPONENT for term in polynomial.polynomial.terms):
        raise ValueError(f"curve polynomial exceeds total degree {_MAX_EXPONENT}")


class AffineCurveRequest(StrictModel):
    """An affine plane curve ``f(x, y) = 0`` over ``QQ``."""

    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_affine_plane(self) -> Self:
        _require_curve_polynomial(self.polynomial)
        if len(self.polynomial.variables) != 2:
            raise ValueError("affine plane curves require exactly two variables")
        return self


class ProjectiveClosureRequest(StrictModel):
    """Homogenize an affine plane curve with the reserved coordinate ``z``."""

    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_available_homogenizing_coordinate(self) -> Self:
        _require_curve_polynomial(self.polynomial)
        if len(self.polynomial.variables) != 2:
            raise ValueError("projective closure requires exactly two variables")
        if HOMOGENIZING_COORDINATE in self.polynomial.variables:
            raise ValueError(
                "affine variable axis must not contain the reserved "
                f"homogenizing coordinate {HOMOGENIZING_COORDINATE!r}"
            )
        return self


class AffineChartRequest(StrictModel):
    """Dehomogenize a homogeneous projective plane curve on one chart."""

    polynomial: RationalPolynomial
    chart_variable: PolynomialVariable

    @model_validator(mode="after")
    def require_homogeneous_projective_plane(self) -> Self:
        _require_curve_polynomial(self.polynomial)
        if len(self.polynomial.variables) != 3:
            raise ValueError("projective plane curves require exactly three variables")
        if self.chart_variable not in self.polynomial.variables:
            raise ValueError("chart_variable must belong to the polynomial axis")
        if not rational_polynomial_to_sympy(self.polynomial).is_homogeneous:
            raise ValueError("projective polynomial must be homogeneous")
        return self


class AffineCurveResult(StrictModel):
    is_valid: bool
    degree: int = Field(ge=0, le=_MAX_EXPONENT)
    method: Literal["SYMPY_CURVE_CHECK"] = "SYMPY_CURVE_CHECK"


class ProjectiveClosureResult(StrictModel):
    polynomial: RationalPolynomial
    method: Literal["HOMOGENIZATION"] = "HOMOGENIZATION"


class AffineChartResult(StrictModel):
    polynomial: RationalPolynomial
    method: Literal["DEHOMOGENIZATION"] = "DEHOMOGENIZATION"


__all__ = [
    "MAX_VARS",
    "AffineChartRequest",
    "AffineChartResult",
    "AffineCurveRequest",
    "AffineCurveResult",
    "ProjectiveClosureRequest",
    "ProjectiveClosureResult",
]
