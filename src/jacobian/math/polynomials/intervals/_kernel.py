"""Exact natural interval extension for sparse rational polynomials."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
)

type _FractionInterval = tuple[Fraction, Fraction]


def term_is_zero_on_box(
    term: RationalPolynomialTerm,
    box: RationalBox,
) -> bool:
    """Return whether one monomial vanishes identically on the whole box."""

    return any(
        exponent > 0 and interval.lower.num == "0" and interval.upper.num == "0"
        for exponent, interval in zip(term.exponents, box.intervals, strict=True)
    )


def _multiply_intervals(
    left: _FractionInterval,
    right: _FractionInterval,
) -> _FractionInterval:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def _power_interval(value: _FractionInterval, exponent: int) -> _FractionInterval:
    if exponent == 0:
        return Fraction(1), Fraction(1)
    lower, upper = value
    if exponent % 2 == 1:
        return lower**exponent, upper**exponent
    if lower >= 0:
        return lower**exponent, upper**exponent
    if upper <= 0:
        return upper**exponent, lower**exponent
    return Fraction(0), max(lower**exponent, upper**exponent)


def _scale_interval(value: _FractionInterval, scalar: Fraction) -> _FractionInterval:
    if scalar >= 0:
        return value[0] * scalar, value[1] * scalar
    return value[1] * scalar, value[0] * scalar


def natural_interval_extension(
    polynomial: RationalPolynomial,
    box: RationalBox,
) -> ClosedRationalInterval:
    """Evaluate the canonical sparse form by exact rational interval arithmetic."""

    coordinates = tuple(
        (interval.lower.as_fraction(), interval.upper.as_fraction())
        for interval in box.intervals
    )
    total: _FractionInterval | None = None
    for term in polynomial.polynomial.terms:
        if term_is_zero_on_box(term, box):
            continue
        monomial: _FractionInterval = (Fraction(1), Fraction(1))
        for coordinate, exponent in zip(coordinates, term.exponents, strict=True):
            if exponent:
                monomial = _multiply_intervals(
                    monomial,
                    _power_interval(coordinate, exponent),
                )
        term_interval = _scale_interval(
            monomial,
            term.coefficient.as_fraction(),
        )
        total = (
            term_interval
            if total is None
            else (total[0] + term_interval[0], total[1] + term_interval[1])
        )

    lower, upper = total or (Fraction(0), Fraction(0))
    return ClosedRationalInterval(
        lower=CanonicalRational.from_fraction(lower),
        upper=CanonicalRational.from_fraction(upper),
    )


__all__ = ["natural_interval_extension", "term_is_zero_on_box"]
