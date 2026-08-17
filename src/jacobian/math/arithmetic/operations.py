"""Exact arithmetic on Python integers and fractions."""

from collections.abc import Iterable
from fractions import Fraction
from math import gcd, lcm
from typing import SupportsIndex

__all__ = [
    "absolute_value",
    "integerize_rational_vector",
    "primitive_integer_vector",
    "quotient",
    "reciprocal",
    "sign",
    "sum_rationals",
]


def absolute_value(value: SupportsIndex) -> int:
    """Return the exact absolute value of an integer-like value."""

    return abs(value.__index__())


def sign(value: SupportsIndex) -> int:
    """Return -1, 0, or 1 according to the sign of an integer."""

    integer = value.__index__()
    return (integer > 0) - (integer < 0)


def reciprocal(value: Fraction | int) -> Fraction:
    """Return the exact reciprocal, rejecting zero."""

    rational = Fraction(value)
    if not rational:
        raise ZeroDivisionError("zero has no reciprocal")
    return 1 / rational


def sum_rationals(left: Fraction | int, right: Fraction | int) -> Fraction:
    """Add two exact rational values."""

    return Fraction(left) + Fraction(right)


def integerize_rational_vector(values: Iterable[Fraction | int]) -> tuple[int, ...]:
    """Scale exact rationals to integer coordinates with a shared denominator."""

    rationals = tuple(Fraction(value) for value in values)
    common_denominator = lcm(*(value.denominator for value in rationals))
    return tuple(
        value.numerator * (common_denominator // value.denominator)
        for value in rationals
    )


def primitive_integer_vector(values: Iterable[Fraction | int]) -> tuple[int, ...]:
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


def quotient(left: Fraction | int, right: Fraction | int) -> Fraction:
    """Divide two exact rational values."""

    divisor = Fraction(right)
    if not divisor:
        raise ZeroDivisionError("division by zero")
    return Fraction(left) / divisor
