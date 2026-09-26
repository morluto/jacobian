"""Exact rational unit-quaternion values and operation request models."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers

MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS = 512
"""Maximum decimal digits in a rational coordinate numerator or denominator."""

QuaternionCoordinates = Annotated[
    tuple[CanonicalRational, CanonicalRational, CanonicalRational, CanonicalRational],
    Field(min_length=4, max_length=4),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quaternion.rational_unit.{reason}", message)


def _integer_digit_count(value: int) -> int:
    """Count digits without stringifying oversized authored integers."""
    magnitude = abs(value)
    if (
        magnitude.bit_length() > 3 * MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS
        and magnitude >= 10**MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS
    ):
        return MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS + 1
    return len(str(magnitude))


def _component_is_canonical(value: object) -> bool:
    if not isinstance(value, CanonicalRational):
        return False
    numerator = getattr(value, "num", None)
    denominator = getattr(value, "den", None)
    return (
        type(numerator) is int
        and type(denominator) is int
        and denominator > 0
        and gcd(abs(numerator), denominator) == 1
        and (numerator != 0 or denominator == 1)
    )


def _raw_integer_too_wide(value: object) -> bool:
    if type(value) is int:
        return (
            _integer_digit_count(value) > MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS
        )
    if isinstance(value, str):
        digits = len(value) - (1 if value.startswith("-") else 0)
        return digits > MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS
    return False


def _preflight_raw_coordinates(data: object) -> object:
    if not isinstance(data, dict):
        return data
    coordinates = data.get("coordinates")
    if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 4:
        return data
    for coordinate in coordinates:
        if isinstance(coordinate, CanonicalRational):
            if not _component_is_canonical(coordinate):
                raise _validation_error(
                    "rational_coordinate",
                    "each coordinate must be a canonical reduced rational",
                )
            if (
                max(
                    _integer_digit_count(coordinate.num),
                    _integer_digit_count(coordinate.den),
                )
                > MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS
            ):
                raise _validation_error(
                    "component_digits",
                    "coordinate numerator and denominator admit at most "
                    f"{MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS} digits",
                )
            continue
        if isinstance(coordinate, dict) and (
            _raw_integer_too_wide(coordinate.get("num"))
            or _raw_integer_too_wide(coordinate.get("den"))
        ):
            raise _validation_error(
                "component_digits",
                "coordinate numerator and denominator admit at most "
                f"{MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS} digits",
            )
    return data


class RationalUnitQuaternion(StrictModel):
    """A canonical unit quaternion ``a + b*i + c*j + d*k`` over QQ.

    The tuple order is scalar, ``i``, ``j``, ``k``. Each rational is reduced
    with positive denominator, and the exact sum of coordinate squares is one.
    This value denotes an element of the rational unit-quaternion group; general
    nonunit quaternions are outside this contract.
    """

    coordinates: QuaternionCoordinates

    @model_validator(mode="before")
    @classmethod
    def preflight_coordinates(cls, data: object) -> object:
        return _preflight_raw_coordinates(canonicalize_json_containers(data))

    @model_validator(mode="after")
    def require_unit_norm(self) -> Self:
        if any(not _component_is_canonical(value) for value in self.coordinates):
            raise _validation_error(
                "rational_coordinate",
                "each coordinate must be a canonical reduced rational",
            )
        if any(
            max(_integer_digit_count(value.num), _integer_digit_count(value.den))
            > MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS
            for value in self.coordinates
        ):
            raise _validation_error(
                "component_digits",
                "coordinate numerator and denominator admit at most "
                f"{MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS} digits",
            )
        norm_squared = sum(
            (coordinate.as_fraction() ** 2 for coordinate in self.coordinates),
            Fraction(0),
        )
        if norm_squared != 1:
            raise _validation_error(
                "norm_one", "rational unit quaternion coordinates must have norm one"
            )
        return self

    @classmethod
    def _from_kernel(cls, coordinates: QuaternionCoordinates) -> Self:
        """Build a trusted multiplication result without replaying norm arithmetic."""
        return cls.model_construct(coordinates=coordinates)


class RationalUnitQuaternionBinaryRequest(StrictModel):
    """Two exact rational unit quaternions for group multiplication."""

    left: RationalUnitQuaternion
    right: RationalUnitQuaternion


class RationalUnitQuaternionUnaryRequest(StrictModel):
    """One exact rational unit quaternion for a unary group operation."""

    value: RationalUnitQuaternion


__all__ = [
    "MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS",
    "QuaternionCoordinates",
    "RationalUnitQuaternion",
    "RationalUnitQuaternionBinaryRequest",
    "RationalUnitQuaternionUnaryRequest",
]
