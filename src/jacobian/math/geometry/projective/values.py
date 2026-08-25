"""Semantic values for rational projective geometry."""

from __future__ import annotations

from math import gcd, lcm
from typing import Annotated, Self

from pydantic import StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer


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


__all__ = ["PrimitiveProjectiveTriple", "RationalProjectiveLine"]
