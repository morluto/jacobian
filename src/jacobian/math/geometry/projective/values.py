"""Semantic values for rational projective geometry."""

from __future__ import annotations

from math import gcd, lcm
from typing import Annotated, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.number_fields.values import (
    NumberFieldEmbedding,
    SimpleNumberFieldElement,
)
from jacobian.math.polynomials.values import PolynomialVariable


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


ProjectiveLabel = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$",
        strict=True,
    ),
]


def _primitive_integer_triple(
    coefficients: tuple[CanonicalRational, CanonicalRational, CanonicalRational],
) -> tuple[int, int, int]:
    fractions = tuple(coefficient.as_fraction() for coefficient in coefficients)
    common_denominator = lcm(*(coefficient.denominator for coefficient in fractions))
    integers = tuple(
        coefficient.numerator * (common_denominator // coefficient.denominator)
        for coefficient in fractions
    )
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    if divisor == 0:
        raise _validation_error(
            "a_projective_line_coefficient_triple_nonzero",
            "a projective line coefficient triple must be nonzero",
        )
    primitive = tuple(value // divisor for value in integers)
    if next(value for value in primitive if value) < 0:
        primitive = tuple(-value for value in primitive)
    return (primitive[0], primitive[1], primitive[2])


class RationalProjectiveLine(StrictModel):
    """One labelled line ``a*x + b*y + c*z = 0`` over QQ."""

    label: ProjectiveLabel
    coefficients: tuple[
        CanonicalRational,
        CanonicalRational,
        CanonicalRational,
    ]

    @model_validator(mode="after")
    def require_nonzero_line(self) -> Self:
        _primitive_integer_triple(self.coefficients)
        return self


class PrimitiveProjectiveTriple(StrictModel):
    """Canonical primitive integer homogeneous coordinates."""

    coordinates: tuple[str, str, str]

    @model_validator(mode="after")
    def require_canonical_primitive_coordinates(self) -> Self:
        try:
            values = tuple(parse_canonical_integer(value) for value in self.coordinates)
        except ValueError as exc:
            raise _validation_error(
                "projective_coordinates_integer_strings",
                "projective coordinates must be integer strings",
            ) from exc
        if (
            tuple(format_canonical_integer(value) for value in values)
            != self.coordinates
        ):
            raise _validation_error(
                "projective_coordinates_canonical_integer_strings",
                "projective coordinates must be canonical integer strings",
            )
        divisor = 0
        for value in values:
            divisor = gcd(divisor, abs(value))
        if divisor != 1:
            raise _validation_error(
                "projective_coordinates_nonzero_primitive",
                "projective coordinates must be nonzero and primitive",
            )
        if next(value for value in values if value) < 0:
            raise _validation_error(
                "first_nonzero_projective_coordinate_positive",
                "the first nonzero projective coordinate must be positive",
            )
        return self


class AlgebraicProjectivePlanePoint(StrictModel):
    """A canonical projective-plane point in one exact embedded number field.

    Coordinates use the field presentation's reduced power basis.  The selected
    embedding supplies the geometric point over ``QQbar``.  Normalization uses
    the first nonzero projective coordinate, which is exactly one; this makes
    chart membership and equality independent of a backend's root objects.
    """

    axis: tuple[
        PolynomialVariable,
        PolynomialVariable,
        PolynomialVariable,
    ]
    embedding: NumberFieldEmbedding
    coordinates: tuple[
        SimpleNumberFieldElement,
        SimpleNumberFieldElement,
        SimpleNumberFieldElement,
    ]
    chart_index: StrictInt = Field(ge=0, le=2)

    @model_validator(mode="after")
    def bind_parent_and_normalization(self) -> Self:
        if len(set(self.axis)) != 3:
            raise _validation_error(
                "projective_plane_axis",
                "an algebraic projective-plane point needs three distinct axes",
            )
        presentation = self.embedding.presentation
        if any(
            coordinate.presentation != presentation for coordinate in self.coordinates
        ):
            raise _validation_error(
                "projective_point_field",
                "all projective coordinates and the selected embedding must share one field presentation",
            )
        zero = tuple(
            coefficient.as_fraction() == 0
            for coordinate in self.coordinates
            for coefficient in coordinate.coefficients_ascending
        )
        degree = presentation.degree
        coordinate_is_zero = tuple(
            all(zero[index * degree : (index + 1) * degree]) for index in range(3)
        )
        coordinate_is_one = tuple(
            coordinate.coefficients_ascending[0].as_fraction() == 1
            and all(
                coefficient.as_fraction() == 0
                for coefficient in coordinate.coefficients_ascending[1:]
            )
            for coordinate in self.coordinates
        )
        if any(not coordinate_is_zero[index] for index in range(self.chart_index)):
            raise _validation_error(
                "projective_point_chart",
                "coordinates before the declared projective chart must be zero",
            )
        if not coordinate_is_one[self.chart_index]:
            raise _validation_error(
                "projective_point_normalization",
                "the declared projective chart coordinate must be exactly one",
            )
        return self


__all__ = [
    "AlgebraicProjectivePlanePoint",
    "PrimitiveProjectiveTriple",
    "RationalProjectiveLine",
]
