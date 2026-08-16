"""Behavioral tests for the CanonicalRational bidirectional conversion boundary.

These tests exercise the wire-to-mathematics and mathematics-to-wire conversion
methods above CPython's 4,300-digit integer string conversion limit, confirming
that the boundary is safe for the full 32,768-digit contract range.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.contracts.exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)

# Values with more than 4,300 digits — above CPython's default
# PYTHONINTMAXSTRDIGITS limit but well within the 32,768-digit contract ceiling.
_LARGE_NUMERATOR = 10**5000 + 3  # 5,001-digit value, not a power of 10
_LARGE_DENOMINATOR = 10**5000 + 7  # 5,001-digit value, coprime to numerator


def test_as_integer_ratio_round_trips_small_values() -> None:
    value = CanonicalRational(num="3", den="7")
    numerator, denominator = value.as_integer_ratio()

    assert numerator == 3
    assert denominator == 7
    assert value.as_fraction() == Fraction(numerator, denominator)


def test_from_integer_ratio_constructs_reduced_rational() -> None:
    value = CanonicalRational.from_integer_ratio(6, 4)

    assert value.num == "3"
    assert value.den == "2"


def test_from_fraction_constructs_reduced_rational() -> None:
    value = CanonicalRational.from_fraction(Fraction(6, 4))

    assert value.num == "3"
    assert value.den == "2"


def test_from_integer_ratio_preserves_negative_numerator() -> None:
    value = CanonicalRational.from_integer_ratio(-9, 3)

    assert value.num == "-3"
    assert value.den == "1"


def test_from_integer_ratio_normalizes_negative_denominator() -> None:
    value = CanonicalRational.from_integer_ratio(5, -3)

    assert value.num == "-5"
    assert value.den == "3"


def test_from_integer_ratio_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="rational denominator cannot be zero"):
        CanonicalRational.from_integer_ratio(1, 0)


def test_as_integer_ratio_above_digit_limit() -> None:
    value = CanonicalRational(
        num="1" + "0" * 5000,
        den="1",
    )
    numerator, denominator = value.as_integer_ratio()

    assert numerator == 10**5000
    assert denominator == 1


def test_from_integer_ratio_above_digit_limit() -> None:
    value = CanonicalRational.from_integer_ratio(_LARGE_NUMERATOR, 1)

    assert len(value.num.lstrip("-")) > 4300
    assert value.den == "1"
    assert value.as_integer_ratio() == (_LARGE_NUMERATOR, 1)


def test_from_fraction_above_digit_limit() -> None:
    fraction = Fraction(_LARGE_NUMERATOR, _LARGE_DENOMINATOR)
    value = CanonicalRational.from_fraction(fraction)

    assert len(value.num.lstrip("-")) > 4300
    assert len(value.den.lstrip("-")) > 4300
    assert value.as_fraction() == fraction


def test_round_trip_above_digit_limit() -> None:
    fraction = Fraction(_LARGE_NUMERATOR, _LARGE_DENOMINATOR)
    value = CanonicalRational.from_fraction(fraction)

    assert CanonicalRational.from_fraction(value.as_fraction()) == value
    assert value.as_integer_ratio() == (fraction.numerator, fraction.denominator)


def test_from_integer_ratio_rejects_above_contract_digit_limit() -> None:
    too_large = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS + 1)

    with pytest.raises(ValidationError, match=r"exceed the canonical"):
        CanonicalRational.from_integer_ratio(too_large, 1)


def test_from_fraction_rejects_above_contract_digit_limit() -> None:
    too_large = Fraction(10 ** (MAX_CANONICAL_RATIONAL_DIGITS + 1), 1)

    with pytest.raises(ValidationError, match=r"exceed the canonical"):
        CanonicalRational.from_fraction(too_large)
