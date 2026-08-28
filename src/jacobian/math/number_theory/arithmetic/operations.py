"""Exact arithmetic on Python integers and fractions."""

from collections.abc import Iterable
from fractions import Fraction
from math import gcd, lcm
from typing import SupportsIndex

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.arithmetic.values import IntegerValue

__all__ = [
    "absolute_value",
    "integerize_rational_vector",
    "primitive_integer_vector",
    "quotient",
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
        raise ZeroDivisionError("zero has no reciprocal")
    return 1 / rational


def sum_rationals(
    left: Fraction | int | IntegerValue, right: Fraction | int | IntegerValue
) -> Fraction:
    """Add two exact rational values."""

    return _as_rational(left) + _as_rational(right)


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
        raise ZeroDivisionError("division by zero")
    return _as_rational(left) / divisor
