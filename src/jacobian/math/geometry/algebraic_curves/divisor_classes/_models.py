"""Typed contract for plane-curve strict-transform divisor classes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.geometry.blowup_p2._models import BlowupP2Surface
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
)

MAX_CURVE_DIVISOR_DEGREE = 12
MAX_CURVE_DIVISOR_TERMS = 64
MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS = 32
MAX_CURVE_DIVISOR_POINT_DIGITS = 16
MAX_CURVE_DIVISOR_WORK = 1_000_000
MAX_CURVE_DIVISOR_INTERMEDIATE_DIGITS = 32_768
MAX_CURVE_DIVISOR_ALLOCATION_UNITS = 512


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"plane_curve_divisor.{reason}", message)


def _admit_raw_surface(surface: object) -> None:
    if not isinstance(surface, Mapping):
        return
    points = surface.get("points")
    if not isinstance(points, (list, tuple)):
        return
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
        _admit_raw_surface(data.get("surface"))
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_plane_curve_and_axis_map(self) -> Self:
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
        if set(self.projective_coordinate_variables) != set(self.polynomial.variables):
            raise _error(
                "coordinate_transport",
                "projective coordinate variables must be a permutation of the polynomial axis",
            )
        return self


__all__ = [
    "MAX_CURVE_DIVISOR_ALLOCATION_UNITS",
    "MAX_CURVE_DIVISOR_COEFFICIENT_DIGITS",
    "MAX_CURVE_DIVISOR_DEGREE",
    "MAX_CURVE_DIVISOR_INTERMEDIATE_DIGITS",
    "MAX_CURVE_DIVISOR_POINT_DIGITS",
    "MAX_CURVE_DIVISOR_TERMS",
    "MAX_CURVE_DIVISOR_WORK",
    "PlaneCurveStrictTransformRequest",
]
