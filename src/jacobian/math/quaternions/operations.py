"""Bounded exact group operations for rational unit quaternions."""

from __future__ import annotations

from fractions import Fraction
from typing import NoReturn, cast

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.quaternions._models import (
    MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS,
    QuaternionCoordinates,
    RationalUnitQuaternion,
    _component_is_canonical,
    _integer_digit_count,
)

MAX_RATIONAL_UNIT_QUATERNION_MULTIPLY_WORK = 20_000_000
"""Maximum admitted decimal digit-square units for one Hamilton product."""


def _domain_error(location: str, reason: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,),
        code=f"quaternion.rational_unit.{reason}",
        message=message,
    )


def _resource_error(location: str, reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=(location,),
        code=f"quaternion.rational_unit.{reason}",
        message=message,
    )


def _admit_operand(value: object, location: str = "value") -> RationalUnitQuaternion:
    if not isinstance(value, RationalUnitQuaternion):
        _domain_error(
            location, "operand_type", "expected a RationalUnitQuaternion value"
        )
    coordinates = getattr(value, "coordinates", None)
    if not isinstance(coordinates, tuple) or len(coordinates) != 4:
        _domain_error(
            location, "coordinate_shape", "quaternion requires four coordinates"
        )
    for coordinate in coordinates:
        if not _component_is_canonical(coordinate):
            _domain_error(
                location,
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
            _resource_error(
                location,
                "component_digits_exceed_envelope",
                "coordinate numerator and denominator admit at most "
                f"{MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS} digits",
            )
    norm_squared = sum(
        (coordinate.as_fraction() ** 2 for coordinate in coordinates), Fraction(0)
    )
    if norm_squared != 1:
        _domain_error(
            location, "norm_one", "quaternion operand must have exact norm one"
        )
    return value


def _fraction(value: CanonicalRational) -> Fraction:
    return value.as_fraction()


def _admit_product_work(
    left: RationalUnitQuaternion, right: RationalUnitQuaternion
) -> None:
    left_digits = max(
        max(_integer_digit_count(value.num), _integer_digit_count(value.den))
        for value in left.coordinates
    )
    right_digits = max(
        max(_integer_digit_count(value.num), _integer_digit_count(value.den))
        for value in right.coordinates
    )
    # Four output coordinates each sum at most four rational products. The
    # factor 64 covers those 16 products and the exact gcd/addition work.
    # Every admitted operand is already bounded at 512 digits, so unreduced
    # intermediate denominators are bounded polynomially and exact cancellations
    # never expand past that ceiling; the reduced output itself is admitted
    # after canonicalization below.
    work = 64 * left_digits * right_digits
    if work > MAX_RATIONAL_UNIT_QUATERNION_MULTIPLY_WORK:
        _resource_error(
            "operands",
            "multiply_work_exceeds_envelope",
            "exact quaternion multiplication exceeds its 20,000,000-unit work envelope",
        )


def _admit_reduced_output(value: Fraction) -> None:
    if (
        max(
            _integer_digit_count(value.numerator),
            _integer_digit_count(value.denominator),
        )
        > MAX_RATIONAL_UNIT_QUATERNION_COMPONENT_DIGITS
    ):
        _resource_error(
            "result",
            "product_growth_exceeds_envelope",
            "exact quaternion product coordinate exceeds the 512-digit envelope",
        )


def _trusted_rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.model_construct(num=value.numerator, den=value.denominator)


def multiply_rational_unit_quaternions(
    left: RationalUnitQuaternion, right: RationalUnitQuaternion
) -> RationalUnitQuaternion:
    """Return the exact Hamilton product, with digit and work admission first."""
    left = _admit_operand(left, "left")
    right = _admit_operand(right, "right")
    _admit_product_work(left, right)
    a, b, c, d = tuple(_fraction(value) for value in left.coordinates)
    e, f, g, h = tuple(_fraction(value) for value in right.coordinates)
    product = (
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    )
    for value in product:
        _admit_reduced_output(value)
    coordinates = cast(
        QuaternionCoordinates, tuple(_trusted_rational(value) for value in product)
    )
    return RationalUnitQuaternion._from_kernel(coordinates)


def conjugate_rational_unit_quaternion(
    value: RationalUnitQuaternion,
) -> RationalUnitQuaternion:
    """Return the exact quaternion conjugate ``(a,-b,-c,-d)``."""
    value = _admit_operand(value)
    coordinates = cast(
        QuaternionCoordinates,
        tuple(
            _trusted_rational(
                _fraction(coordinate) if index == 0 else -_fraction(coordinate)
            )
            for index, coordinate in enumerate(value.coordinates)
        ),
    )
    return RationalUnitQuaternion._from_kernel(coordinates)


def inverse_rational_unit_quaternion(
    value: RationalUnitQuaternion,
) -> RationalUnitQuaternion:
    """Return the group inverse, equal to conjugation for norm-one values."""
    return conjugate_rational_unit_quaternion(value)


def rational_unit_quaternion_scalar_part(
    value: RationalUnitQuaternion,
) -> CanonicalRational:
    """Return the exact rational scalar coordinate ``a``."""
    value = _admit_operand(value)
    return value.coordinates[0]


__all__ = [
    "MAX_RATIONAL_UNIT_QUATERNION_MULTIPLY_WORK",
    "conjugate_rational_unit_quaternion",
    "inverse_rational_unit_quaternion",
    "multiply_rational_unit_quaternions",
    "rational_unit_quaternion_scalar_part",
]
