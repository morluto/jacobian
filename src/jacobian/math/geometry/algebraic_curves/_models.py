"""Typed wire contracts for plane algebraic curve operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.maps._models import VariablePoint
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalFunction,
    RationalPolynomial,
    require_polynomial_budget,
    require_sparse_polynomial_budget,
)

MAX_VARS = 3
HOMOGENIZING_COORDINATE = "z"
_MAX_CURVE_TERMS = 256
_MAX_CURVE_EXPONENT = 64
_MAX_CURVE_COEFFICIENT_DIGITS = 128


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"plane_algebraic_curve.{reason}", message)


def _validation_error_from(exc: ValueError) -> PydanticCustomError:
    """Translate owner-kernel admission failures at the public model boundary."""

    message = str(exc)
    reasons = (
        ("point must lie", "point_not_on_conic"),
        ("smooth irreducible", "conic_not_smooth_irreducible"),
        ("degree exactly two", "conic_degree_invalid"),
        ("exactly two variables", "conic_axis_invalid"),
        ("parameter must be distinct", "parameter_axis_collision"),
        ("complete ordered axis", "point_axis_mismatch"),
        ("128-digit", "coefficient_height_exceeded"),
        ("output exceeds", "result_height_exceeded"),
        ("gradient-normalized", "parametrization_not_canonical"),
        ("source conic identity", "parametrization_identity_invalid"),
        ("projective parameter infinity", "exceptional_point_invalid"),
        ("quadratic denominator", "pencil_denominator_invalid"),
        ("finite parameter", "inverse_chart_invalid"),
        ("tangent line", "inverse_denominator_invalid"),
        ("base point", "projective_base_point"),
        ("source closure", "projective_closure_invalid"),
        ("chart coordinate", "projective_chart_invalid"),
    )
    for fragment, reason in reasons:
        if fragment in message:
            return _validation_error(reason, message)
    return _validation_error("admission_failed", message)


def _require_curve_polynomial(polynomial: RationalPolynomial) -> None:
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_CURVE_TERMS,
        maximum_exponent=_MAX_CURVE_EXPONENT,
        maximum_coefficient_digits=_MAX_CURVE_COEFFICIENT_DIGITS,
        label="curve polynomial",
    )
    if any(
        sum(term.exponents) > _MAX_CURVE_EXPONENT
        for term in polynomial.polynomial.terms
    ):
        raise ValueError(f"curve polynomial exceeds total degree {_MAX_CURVE_EXPONENT}")


class AffineCurveRequest(StrictModel):
    """An affine plane curve ``f(x, y) = 0`` over ``QQ``."""

    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_affine_plane(self) -> Self:
        try:
            _require_curve_polynomial(self.polynomial)
        except ValueError as exc:
            raise _validation_error_from(exc) from exc
        if len(self.polynomial.variables) != 2:
            raise _validation_error(
                "affine_axis_invalid",
                "affine plane curves require exactly two variables",
            )
        return self


class ProjectiveClosureRequest(StrictModel):
    """Homogenize an affine plane curve with the reserved coordinate ``z``."""

    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_available_homogenizing_coordinate(self) -> Self:
        try:
            _require_curve_polynomial(self.polynomial)
        except ValueError as exc:
            raise _validation_error_from(exc) from exc
        if len(self.polynomial.variables) != 2:
            raise _validation_error(
                "closure_axis_invalid",
                "projective closure requires exactly two variables",
            )
        if HOMOGENIZING_COORDINATE in self.polynomial.variables:
            raise _validation_error(
                "homogenizing_coordinate_reserved",
                "affine variable axis must not contain the reserved "
                f"homogenizing coordinate {HOMOGENIZING_COORDINATE!r}",
            )
        return self


class AffineChartRequest(StrictModel):
    """Dehomogenize a homogeneous projective plane curve on one chart."""

    polynomial: RationalPolynomial
    chart_variable: PolynomialVariable

    @model_validator(mode="after")
    def require_homogeneous_projective_plane(self) -> Self:
        try:
            _require_curve_polynomial(self.polynomial)
        except ValueError as exc:
            raise _validation_error_from(exc) from exc
        if len(self.polynomial.variables) != 3:
            raise _validation_error(
                "chart_axis_invalid",
                "projective plane curves require exactly three variables",
            )
        if self.chart_variable not in self.polynomial.variables:
            raise _validation_error(
                "chart_variable_axis_mismatch",
                "chart_variable must belong to the polynomial axis",
            )
        if not rational_polynomial_to_sympy(self.polynomial).is_homogeneous:
            raise _validation_error(
                "polynomial_not_homogeneous",
                "projective polynomial must be homogeneous",
            )
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


class AffineCurveResult(StrictModel):
    is_valid: bool
    degree: int = Field(ge=0, le=_MAX_CURVE_EXPONENT)
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
    def require_structural_contract(self) -> Self:
        """Check only bounded wire-shape relations between result fields.

        The admitted line-pencil kernel establishes the parametrization
        identities. Parsing a serialized result must not reconstruct and replay
        that computation; independently supplied claims are outside this result
        contract.
        """
        try:
            require_polynomial_budget(
                self.source_polynomial,
                maximum_terms=6,
                maximum_exponent=2,
                maximum_coefficient_digits=_MAX_CURVE_COEFFICIENT_DIGITS,
                label="source conic",
            )
            require_polynomial_budget(
                self.finite_parameter_denominator,
                maximum_terms=3,
                maximum_exponent=2,
                maximum_coefficient_digits=_MAX_CURVE_COEFFICIENT_DIGITS,
                label="finite-parameter denominator",
            )
            for label, function in (
                ("first parametrization coordinate", self.coordinates[0]),
                ("second parametrization coordinate", self.coordinates[1]),
                ("inverse parameter", self.inverse_parameter),
            ):
                require_sparse_polynomial_budget(
                    function.numerator,
                    maximum_terms=3,
                    maximum_exponent=2,
                    maximum_coefficient_digits=_MAX_CURVE_COEFFICIENT_DIGITS,
                    label=f"{label} numerator",
                )
                require_sparse_polynomial_budget(
                    function.denominator,
                    maximum_terms=3,
                    maximum_exponent=2,
                    maximum_coefficient_digits=_MAX_CURVE_COEFFICIENT_DIGITS,
                    label=f"{label} denominator",
                )
            for coordinate in self.exceptional_point.values:
                require_bounded_rational(
                    coordinate,
                    max_digits=_MAX_CURVE_COEFFICIENT_DIGITS,
                    label="exceptional point coordinate",
                )
        except ValueError as exc:
            raise _validation_error_from(exc) from exc
        if len(self.source_polynomial.variables) != 2:
            raise _validation_error(
                "conic_axis_invalid",
                "rational conic parametrization requires exactly two variables",
            )
        if self.exceptional_point.variables != self.source_polynomial.variables:
            raise _validation_error(
                "point_axis_mismatch",
                "conic point must use the polynomial's complete ordered axis",
            )
        if self.parameter in self.source_polynomial.variables:
            raise _validation_error(
                "parameter_axis_collision",
                "parameter must be distinct from both conic variables",
            )
        if any(
            coordinate.variables != (self.parameter,) for coordinate in self.coordinates
        ):
            raise _validation_error(
                "coordinate_axis_invalid",
                "parametrization coordinates must use exactly the parameter axis",
            )
        if self.inverse_parameter.variables != self.source_polynomial.variables:
            raise _validation_error(
                "inverse_axis_invalid",
                "inverse parameter must use the source polynomial's ordered axis",
            )
        if self.finite_parameter_denominator.variables != (self.parameter,):
            raise _validation_error(
                "finite_parameter_axis_invalid",
                "finite-parameter denominator must use exactly the parameter axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: RationalConicParametrizationRequest,
        *,
        coordinates: tuple[RationalFunction, RationalFunction],
        inverse_parameter: RationalFunction,
        finite_parameter_denominator: RationalPolynomial,
    ) -> Self:
        """Build one result after the admitted line-pencil kernel established it."""

        return cls.model_construct(
            source_polynomial=request.polynomial,
            exceptional_point=request.point,
            parameter=request.parameter,
            coordinates=coordinates,
            inverse_parameter=inverse_parameter,
            finite_parameter_denominator=finite_parameter_denominator,
        )


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
