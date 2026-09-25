"""Typed contract for plane-curve strict-transform divisor classes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import decimal_digit_width
from jacobian.math.geometry.blowup_p2._models import BlowupP2Surface
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_CURVE_DIVISOR_DEGREE = 12
MAX_CURVE_DIVISOR_TERMS = 64
MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS = 32
MAX_CURVE_DIVISOR_POINT_DIGITS = 16
MAX_CURVE_DIVISOR_WORK = 1_000_000
MAX_CURVE_DIVISOR_INTERMEDIATE_DIGITS = 32_768
MAX_CURVE_DIVISOR_OUTPUT_BYTES = 250_000


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"plane_curve_divisor.{reason}", message)


def _raw_component_digits(value: object) -> int:
    if isinstance(value, dict):
        return max(
            _raw_component_digits(value.get("num", "0")),
            _raw_component_digits(value.get("den", "1")),
        )
    if isinstance(value, int):
        return decimal_digit_width(abs(value))
    return len(str(value).lstrip("-"))


def _admit_raw_polynomial(polynomial: object) -> None:
    if not isinstance(polynomial, Mapping):
        return
    sparse = polynomial.get("polynomial")
    if not isinstance(sparse, Mapping):
        return
    terms = sparse.get("terms")
    if not isinstance(terms, (list, tuple)):
        return
    if len(terms) > MAX_CURVE_DIVISOR_TERMS:
        raise _error("term_bound", "plane-curve source admits at most 64 terms")
    for term in terms:
        if (
            isinstance(term, Mapping)
            and _raw_component_digits(term.get("coefficient"))
            > MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS
        ):
            raise _error(
                "coefficient_bound",
                "curve coefficients are limited to 32 decimal digits",
            )


def _admit_raw_surface(surface: object) -> None:
    if not isinstance(surface, Mapping):
        return
    points = surface.get("points")
    if not isinstance(points, (list, tuple)):
        return
    if len(points) > 16:
        raise _error("point_bound", "at most 16 blow-up points are admitted")
    for row in points:
        if not isinstance(row, Mapping):
            continue
        point = row.get("point")
        coordinates = point.get("coordinates") if isinstance(point, Mapping) else None
        if not isinstance(coordinates, (list, tuple)):
            continue
        if len(coordinates) != 3:
            raise _error(
                "point_axis", "each point must use three projective coordinates"
            )
        if any(
            _raw_component_digits(value) > MAX_CURVE_DIVISOR_POINT_DIGITS
            for value in coordinates
        ):
            raise _error(
                "point_height",
                "projective point components are limited to 16 decimal digits",
            )


class PlaneCurveStrictTransformRequest(StrictModel):
    """A homogeneous plane curve and a coordinate map into its blow-up parent."""

    polynomial: RationalPolynomial
    surface: BlowupP2Surface
    projective_coordinate_variables: tuple[
        PolynomialVariable, PolynomialVariable, PolynomialVariable
    ] = Field(
        description=(
            "Names of the polynomial variables in the projective point coordinate "
            "order [X0:X1:X2]. This explicit permutation transports the point axis."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def admit_raw_envelopes(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        _admit_raw_polynomial(data.get("polynomial"))
        _admit_raw_surface(data.get("surface"))
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_plane_curve_and_axis_map(self) -> Self:
        try:
            require_polynomial_budget(
                self.polynomial,
                maximum_terms=MAX_CURVE_DIVISOR_TERMS,
                maximum_exponent=MAX_CURVE_DIVISOR_DEGREE,
                maximum_coefficient_digits=MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS,
                label="plane-curve source",
            )
        except ValueError as exc:
            raise _error("source_bound", str(exc)) from exc
        if self.polynomial.domain != "QQ" or len(self.polynomial.variables) != 3:
            raise _error(
                "polynomial_axis", "a plane curve requires three variables over QQ"
            )
        terms = self.polynomial.polynomial.terms
        if not terms:
            raise _error(
                "zero_curve", "the zero polynomial does not define a plane curve"
            )
        degrees = {sum(term.exponents) for term in terms}
        if len(degrees) != 1:
            raise _error("inhomogeneous", "the curve polynomial must be homogeneous")
        degree = next(iter(degrees))
        if not 1 <= degree <= MAX_CURVE_DIVISOR_DEGREE:
            raise _error("degree_bound", "the plane-curve degree must be 1..12")
        if set(self.projective_coordinate_variables) != set(self.polynomial.variables):
            raise _error(
                "coordinate_transport",
                "projective coordinate variables must be a permutation of the polynomial axis",
            )
        if len(self.surface.points) > 16:
            raise _error("point_bound", "at most 16 blow-up points are admitted")
        return self


__all__ = [
    "MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS",
    "MAX_CURVE_DIVISOR_DEGREE",
    "MAX_CURVE_DIVISOR_INTERMEDIATE_DIGITS",
    "MAX_CURVE_DIVISOR_OUTPUT_BYTES",
    "MAX_CURVE_DIVISOR_POINT_DIGITS",
    "MAX_CURVE_DIVISOR_TERMS",
    "MAX_CURVE_DIVISOR_WORK",
    "PlaneCurveStrictTransformRequest",
]
