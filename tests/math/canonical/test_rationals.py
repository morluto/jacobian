"""Behavioral tests for the CanonicalRational bidirectional conversion boundary.

These tests exercise the wire-to-mathematics and mathematics-to-wire conversion
methods above CPython's 4,300-digit integer string conversion limit, confirming
that the boundary is safe for the full 32,768-digit contract range.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
    format_canonical_rational,
)

# Values with more than 4,300 digits — above CPython's default
# PYTHONINTMAXSTRDIGITS limit but well within the 32,768-digit contract ceiling.
_LARGE_NUMERATOR = 10**5000 + 3  # 5,001-digit value, not a power of 10
_LARGE_DENOMINATOR = 10**5000 + 7  # 5,001-digit value, coprime to numerator


def test_as_integer_ratio_round_trips_small_values() -> None:
    value = CanonicalRational(num=3, den=7)
    numerator, denominator = value.as_integer_ratio()

    assert numerator == 3
    assert denominator == 7
    assert value.as_fraction() == Fraction(numerator, denominator)


def test_from_integer_ratio_constructs_reduced_rational() -> None:
    value = CanonicalRational.from_integer_ratio(6, 4)

    assert value.num == 3
    assert value.den == 2


def test_from_fraction_constructs_reduced_rational() -> None:
    value = CanonicalRational.from_fraction(Fraction(6, 4))

    assert value.num == 3
    assert value.den == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Fraction(0), "0"),
        (Fraction(-6, 1), "-6"),
        (Fraction(-6, 8), "-3/4"),
    ],
)
def test_format_canonical_rational(value: Fraction, expected: str) -> None:
    assert format_canonical_rational(value) == expected


def test_canonical_rational_component_digits_uses_canonical_components() -> None:
    assert canonical_rational_component_digits(CanonicalRational(num=-4, den=115)) == 3


def test_from_integer_ratio_preserves_negative_numerator() -> None:
    value = CanonicalRational.from_integer_ratio(-9, 3)

    assert value.num == -3
    assert value.den == 1


def test_from_integer_ratio_normalizes_negative_denominator() -> None:
    value = CanonicalRational.from_integer_ratio(5, -3)

    assert value.num == -5
    assert value.den == 3


def test_from_integer_ratio_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="rational denominator cannot be zero"):
        CanonicalRational.from_integer_ratio(1, 0)


def test_as_integer_ratio_above_digit_limit() -> None:
    value = CanonicalRational(
        num=10**5000,
        den=1,
    )
    numerator, denominator = value.as_integer_ratio()

    assert numerator == 10**5000
    assert denominator == 1


def test_from_integer_ratio_above_digit_limit() -> None:
    value = CanonicalRational.from_integer_ratio(_LARGE_NUMERATOR, 1)

    assert abs(value.num) >= 10**4300
    assert value.den == 1
    assert value.as_integer_ratio() == (_LARGE_NUMERATOR, 1)


def test_from_fraction_above_digit_limit() -> None:
    fraction = Fraction(_LARGE_NUMERATOR, _LARGE_DENOMINATOR)
    value = CanonicalRational.from_fraction(fraction)

    assert abs(value.num) >= 10**4300
    assert value.den >= 10**4300
    assert value.as_fraction() == fraction


def test_round_trip_above_digit_limit() -> None:
    fraction = Fraction(_LARGE_NUMERATOR, _LARGE_DENOMINATOR)
    value = CanonicalRational.from_fraction(fraction)

    assert CanonicalRational.from_fraction(value.as_fraction()) == value
    assert value.as_integer_ratio() == (fraction.numerator, fraction.denominator)
    assert CanonicalRational.model_validate_json(value.model_dump_json()) == value
    assert value.model_dump() == {
        "num": fraction.numerator,
        "den": fraction.denominator,
    }


@pytest.mark.parametrize("component", ["3", True, 3.0])
def test_native_rational_rejects_noninteger_components(component: object) -> None:
    with pytest.raises(ValidationError):
        CanonicalRational.model_validate({"num": component, "den": 7})


@pytest.mark.parametrize(
    "payload",
    [
        '{"num":3,"den":"7"}',
        '{"num":"3","den":7}',
        '{"num":"03","den":"7"}',
        '{"num":"-0","den":"1"}',
        '{"num":"2","den":"4"}',
        '{"num":"1","den":"-2"}',
        '{"num":"0","den":"2"}',
    ],
)
def test_rational_json_requires_canonical_reduced_components(payload: str) -> None:
    with pytest.raises(ValidationError):
        CanonicalRational.model_validate_json(payload)


def test_rational_json_uses_strings_at_every_magnitude() -> None:
    value = CanonicalRational(num=3, den=7)
    assert value.model_dump(mode="json") == {"num": "3", "den": "7"}
    assert CanonicalRational.model_validate_json(value.model_dump_json()) == value


def test_from_integer_ratio_rejects_above_contract_digit_limit() -> None:
    too_large = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS + 1)

    with pytest.raises(ValidationError) as error:
        CanonicalRational.from_integer_ratio(too_large, 1)
    assert error.value.errors()[0]["type"] == "exact_integer.digit_bound"


def test_from_fraction_rejects_above_contract_digit_limit() -> None:
    too_large = Fraction(10 ** (MAX_CANONICAL_RATIONAL_DIGITS + 1), 1)

    with pytest.raises(ValidationError) as error:
        CanonicalRational.from_fraction(too_large)
    assert error.value.errors()[0]["type"] == "exact_integer.digit_bound"
