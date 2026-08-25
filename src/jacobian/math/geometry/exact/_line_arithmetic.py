"""Exact line normalization and distance arithmetic."""

from fractions import Fraction
from math import gcd


def _gcd3(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
    if a == 0 and b == 0 and c == 0:
        return Fraction(0)
    numerators = (a.numerator, b.numerator, c.numerator)
    denominators = (a.denominator, b.denominator, c.denominator)
    common_denominator = 1
    for denominator in denominators:
        common_denominator = (
            common_denominator * denominator // gcd(common_denominator, denominator)
        )
    common_numerator = 0
    for numerator, denominator in zip(numerators, denominators, strict=True):
        common_numerator = gcd(
            common_numerator,
            abs(numerator * (common_denominator // denominator)),
        )
    return Fraction(common_numerator, common_denominator)


def canonical_line_coefficients(
    point: tuple[Fraction, ...], other: tuple[Fraction, ...]
) -> tuple[Fraction, Fraction, Fraction]:
    """Return sign- and gcd-normalized ``(A, B, C)`` for ``Ax + By + C = 0``."""

    dx = other[0] - point[0]
    dy = other[1] - point[1]
    a, b = dy, -dx
    c = -(a * point[0] + b * point[1])
    common = _gcd3(a, b, c)
    if common != 0:
        a, b, c = a / common, b / common, c / common
    for coefficient in (a, b, c):
        if coefficient != 0:
            if coefficient < 0:
                a, b, c = -a, -b, -c
            break
    return a, b, c


def squared_point_line_distance(
    anchor: tuple[Fraction, ...],
    point: tuple[Fraction, ...],
    other: tuple[Fraction, ...],
) -> Fraction:
    """Return the exact squared distance from ``anchor`` to the spanned line."""

    dx = other[0] - point[0]
    dy = other[1] - point[1]
    cross = dx * (anchor[1] - point[1]) - dy * (anchor[0] - point[0])
    return (cross * cross) / (dx * dx + dy * dy)


__all__ = ["canonical_line_coefficients", "squared_point_line_distance"]
