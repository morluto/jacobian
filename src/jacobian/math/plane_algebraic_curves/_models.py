"""Typed wire contracts for plane algebraic curve operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.plane_algebraic_curves._conic import (
    ConicParametrizationData,
    validate_rational_conic_request,
    validate_rational_conic_result_identities,
)
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.maps._models import VariablePoint
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalFunction,
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


class RationalConicParametrizationRequest(StrictModel):
    """A smooth affine rational conic with one supplied rational point."""

    polynomial: RationalPolynomial = Field(
        description=(
            "A total-degree-two polynomial in exactly two ordered variables over "
            "QQ whose projective closure is smooth. Coefficients have at most "
            "128 digits."
        )
    )
    point: VariablePoint = Field(
        description=(
            "A checked rational point on the conic whose complete ordered axis "
            "must exactly match the polynomial variables; each component has at "
            "most 128 digits."
        ),
        examples=[
            {
                "variables": ["x", "y"],
                "values": [
                    {"num": "1", "den": "1"},
                    {"num": "0", "den": "1"},
                ],
            }
        ],
    )
    parameter: PolynomialVariable = Field(
        default="t",
        description=(
            "The rational-function parameter variable; it must differ from both "
            "source variables."
        ),
        examples=["t"],
    )

    @model_validator(mode="after")
    def require_smooth_conic_with_admitted_output(self) -> Self:
        validate_rational_conic_request(self.polynomial, self.point, self.parameter)
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


class RationalConicParametrizationResult(StrictModel):
    """A source-bound affine chart of a smooth projective rational conic."""

    source_polynomial: RationalPolynomial
    exceptional_point: VariablePoint = Field(
        description=(
            "The supplied source point, omitted by the finite affine parameter "
            "chart and attained at projective parameter infinity."
        )
    )
    parameter: PolynomialVariable
    coordinates: tuple[RationalFunction, RationalFunction] = Field(
        description=(
            "Canonical coordinate functions in source-variable order. For every "
            "finite parameter outside finite_parameter_denominator=0 they give "
            "an affine point on the source conic."
        )
    )
    inverse_parameter: RationalFunction = Field(
        description=(
            "The inverse rational parameter in the two source variables. Its "
            "denominator-nonzero chart is the conic minus exceptional_point."
        )
    )
    finite_parameter_denominator: RationalPolynomial = Field(
        description=(
            "The monic quadratic whose finite zeros map to points of the "
            "projective closure outside the source affine chart."
        )
    )
    exceptional_parameter: Literal["PROJECTIVE_INFINITY"] = "PROJECTIVE_INFINITY"
    normalization: Literal["GRADIENT_ORTHOGONAL_LINE_PENCIL"] = (
        "GRADIENT_ORTHOGONAL_LINE_PENCIL"
    )

    @model_validator(mode="after")
    def replay_defining_identities(self) -> Self:
        validate_rational_conic_request(
            self.source_polynomial,
            self.exceptional_point,
            self.parameter,
        )
        validate_rational_conic_result_identities(
            self.source_polynomial,
            self.exceptional_point,
            self.parameter,
            ConicParametrizationData(
                coordinates=self.coordinates,
                inverse_parameter=self.inverse_parameter,
                finite_parameter_denominator=self.finite_parameter_denominator,
            ),
        )
        return self


__all__ = [
    "MAX_VARS",
    "AffineChartRequest",
    "AffineChartResult",
    "AffineCurveRequest",
    "AffineCurveResult",
    "ProjectiveClosureRequest",
    "ProjectiveClosureResult",
    "RationalConicParametrizationRequest",
    "RationalConicParametrizationResult",
]
