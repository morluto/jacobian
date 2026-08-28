"""Exact arithmetic on Python integers and fractions."""

from collections.abc import Iterable
from fractions import Fraction
from math import ceil, floor, gcd, lcm
from typing import SupportsIndex

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.arithmetic.values import IntegerValue

__all__ = [
    "absolute_value",
    "ceiling_rational",
    "continued_fraction",
    "difference_rationals",
    "equal_rationals",
    "floor_rational",
    "integerize_rational_vector",
    "less_than_rationals",
    "maximum_rational",
    "minimum_rational",
    "negate_rational",
    "primitive_integer_vector",
    "product_rationals",
    "quotient",
    "rational_absolute_value",
    "reciprocal",
    "sign",
    "sum_rationals",
]


def _as_python_integer(value: SupportsIndex | IntegerValue) -> int:
    """Return one admitted integer input as its Python integer value."""

    if isinstance(value, IntegerValue):
        return parse_canonical_integer(value.value)
    return value.__index__()


def absolute_value(value: SupportsIndex | IntegerValue) -> IntegerValue:
    """Return the canonical shared integer value of the exact absolute value."""

    return IntegerValue(value=format_canonical_integer(abs(_as_python_integer(value))))


def sign(value: SupportsIndex | IntegerValue) -> int:
    """Return -1, 0, or 1 according to the sign of an integer."""

    integer = _as_python_integer(value)
    return (integer > 0) - (integer < 0)


def _as_rational(value: Fraction | int | IntegerValue) -> Fraction:
    """Return one admitted rational input as its exact Python fraction."""

    if isinstance(value, IntegerValue):
        return Fraction(parse_canonical_integer(value.value))
    return Fraction(value)


def reciprocal(value: Fraction | int | IntegerValue) -> Fraction:
    """Return the exact reciprocal, rejecting zero."""

    rational = _as_rational(value)
    if not rational:
        raise OperationDomainValidationError(
            location=("value",),
            code="arithmetic.reciprocal_requires_nonzero",
            message="reciprocal requires a nonzero rational",
        )
    return 1 / rational


def sum_rationals(
    left: Fraction | int | IntegerValue, right: Fraction | int | IntegerValue
) -> Fraction:
    """Add two exact rational values."""

    return _as_rational(left) + _as_rational(right)


def negate_rational(value: Fraction | int | IntegerValue) -> Fraction:
    """Return the exact additive inverse of a rational value."""

    return -_as_rational(value)


def rational_absolute_value(value: Fraction | int | IntegerValue) -> Fraction:
    """Return the exact absolute value of a rational value."""

    return abs(_as_rational(value))


def difference_rationals(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> Fraction:
    """Subtract two exact rational values."""

    return _as_rational(left) - _as_rational(right)


def product_rationals(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> Fraction:
    """Multiply two exact rational values."""

    return _as_rational(left) * _as_rational(right)


def minimum_rational(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> Fraction:
    """Return the lesser of two exact rational values."""

    return min(_as_rational(left), _as_rational(right))


def maximum_rational(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> Fraction:
    """Return the greater of two exact rational values."""

    return max(_as_rational(left), _as_rational(right))


def floor_rational(value: Fraction | int | IntegerValue) -> int:
    """Return the greatest integer not exceeding an exact rational."""

    return floor(_as_rational(value))


def ceiling_rational(value: Fraction | int | IntegerValue) -> int:
    """Return the least integer not below an exact rational."""

    return ceil(_as_rational(value))


def continued_fraction(
    value: Fraction | int | IntegerValue,
) -> tuple[int, ...]:
    """Return the canonical finite simple continued fraction of a rational."""

    return _continued_fraction_terms(_as_rational(value))


def _continued_fraction_terms(
    rational: Fraction,
    *,
    max_terms: int | None = None,
) -> tuple[int, ...]:
    """Expand one rational, stopping before a bounded result would overflow."""

    numerator = rational.numerator
    denominator = rational.denominator
    terms: list[int] = []
    while denominator:
        quotient, remainder = divmod(numerator, denominator)
        if max_terms is not None and len(terms) == max_terms:
            raise OperationDomainValidationError(
                location=("value",),
                code="arithmetic.continued_fraction_terms_exceed_limit",
                message=(
                    f"continued fraction exceeds the {max_terms}-term result bound"
                ),
            )
        terms.append(quotient)
        numerator, denominator = denominator, remainder
    return tuple(terms)


def equal_rationals(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> bool:
    """Decide exact rational equality."""

    return _as_rational(left) == _as_rational(right)


def less_than_rationals(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> bool:
    """Decide strict order of two exact rational values."""

    return _as_rational(left) < _as_rational(right)


def integerize_rational_vector(
    values: Iterable[Fraction | int | IntegerValue],
) -> tuple[int, ...]:
    """Scale exact rationals to integer coordinates with a shared denominator."""

    rationals = tuple(_as_rational(value) for value in values)
    common_denominator = lcm(*(value.denominator for value in rationals))
    return tuple(
        value.numerator * (common_denominator // value.denominator)
        for value in rationals
    )


def primitive_integer_vector(
    values: Iterable[Fraction | int | IntegerValue],
) -> tuple[int, ...]:
    """Normalize a nonzero rational vector to primitive, positive-leading integers."""

    integers = integerize_rational_vector(values)
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    if not divisor:
        raise ValueError("a primitive integer vector must be nonzero")
    primitive = tuple(value // divisor for value in integers)
    if next(value for value in primitive if value) < 0:
        return tuple(-value for value in primitive)
    return primitive


def quotient(
    left: Fraction | int | IntegerValue, right: Fraction | int | IntegerValue
) -> Fraction:
    """Divide two exact rational values."""

    divisor = _as_rational(right)
    if not divisor:
        raise OperationDomainValidationError(
            location=("right",),
            code="arithmetic.division_requires_nonzero_divisor",
            message="quotient requires a nonzero divisor",
        )
    return _as_rational(left) / divisor
