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
    """One labelled rational homogeneous line carrier."""

    label: ProjectiveLabel
    coefficients: tuple[
        CanonicalRational,
        CanonicalRational,
        CanonicalRational,
    ]

class PrimitiveProjectiveTriple(StrictModel):
    """Integer homogeneous coordinates with canonical decimal spelling."""

    coordinates: tuple[str, str, str]

    @model_validator(mode="after")
    def require_canonical_integer_coordinates(self) -> Self:
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
    def bind_parent_and_field(self) -> Self:
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
        return self


def verify_rational_projective_line(line: RationalProjectiveLine) -> bool:
    """Check that a rational line is a nonzero projective representative."""
    try:
        _primitive_integer_triple(line.coefficients)
    except (TypeError, ValueError):
        return False
    return True


def verify_primitive_projective_triple(triple: PrimitiveProjectiveTriple) -> bool:
    """Check nonzero primitive sign-normalized integer coordinates."""
    try:
        values = tuple(parse_canonical_integer(value) for value in triple.coordinates)
        divisor = 0
        for value in values:
            divisor = gcd(divisor, abs(value))
        return divisor == 1 and next(value for value in values if value) > 0
    except (TypeError, ValueError, StopIteration):
        return False


__all__ = [
    "AlgebraicProjectivePlanePoint",
    "PrimitiveProjectiveTriple",
    "RationalProjectiveLine",
    "verify_primitive_projective_triple",
    "verify_rational_projective_line",
]
