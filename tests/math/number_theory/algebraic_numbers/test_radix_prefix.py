"""Exact bounded radix prefixes of real algebraic values (#2789)."""

from __future__ import annotations

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.algebraic_numbers import _radix_prefix as radix_module
from jacobian.math.number_theory.algebraic_numbers._radix_prefix import (
    MAX_RADIX_PLACES,
    RadixPrefixResult,
    _unique_floor_of_open_interval,
    radix_prefix,
)
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue


def _value(polynomial: tuple[int, ...], root_index: int) -> RealAlgebraicValue:
    return RealAlgebraicValue(polynomial=polynomial, real_root_index=root_index)


def test_sqrt_two_decimal_prefix_matches_known_digits() -> None:
    """sqrt(2) = 1.41421356237309504880...; the selected positive root is index 1."""
    result = radix_prefix(_value((1, 0, -2), 1), 10, 20)
    assert result.integer_part == 1
    assert result.fractional_digits == (
        4,
        1,
        4,
        2,
        1,
        3,
        5,
        6,
        2,
        3,
        7,
        3,
        0,
        9,
        5,
        0,
        4,
        8,
        8,
        0,
    )


def test_sqrt_two_binary_prefix_matches_known_bits() -> None:
    """sqrt(2) = 1.0110101000001001111..._2."""
    result = radix_prefix(_value((1, 0, -2), 1), 2, 16)
    assert result.integer_part == 1
    assert result.fractional_digits == (
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
    )


def test_negative_root_uses_a_signed_integer_part() -> None:
    """The negative root of x^2-2 is -sqrt(2) = -1.41421..."""
    result = radix_prefix(_value((1, 0, -2), 0), 10, 5)
    assert result.integer_part == -2
    # Truncation toward -infinity keeps the nonnegative fractional part.
    assert result.fractional_digits == (5, 8, 5, 7, 8)


def test_rational_root_uses_the_terminating_zero_convention() -> None:
    """1/2 has the prefix 0.50000, not the trailing-(b-1) expansion 0.49999."""
    result = radix_prefix(_value((2, -1), 0), 10, 5)
    assert result.integer_part == 0
    assert result.fractional_digits == (5, 0, 0, 0, 0)
    assert result.convention == "TERMINATING_ZEROS_FOR_RATIONALS"


def test_negative_rational_uses_floor_integer_part() -> None:
    """The returned radix interval must contain a negative rational exactly."""
    result = radix_prefix(_value((2, 1), 0), 10, 3)
    assert result.integer_part == -1
    assert result.fractional_digits == (5, 0, 0)


def test_straddling_isolating_interval_cannot_claim_a_floor() -> None:
    """A nonintegral upper endpoint must not hide a crossed integer."""
    from fractions import Fraction

    assert _unique_floor_of_open_interval(Fraction(7, 5), Fraction(13, 5)) is None
    assert _unique_floor_of_open_interval(Fraction(7, 5), Fraction(9, 5)) == 1


def test_repeating_rational_prefix_is_exact() -> None:
    """1/3 has prefix 0.3333333333 in base ten."""
    result = radix_prefix(_value((3, -1), 0), 10, 10)
    assert result.integer_part == 0
    assert result.fractional_digits == (3,) * 10


def test_zero_fractional_places_returns_the_integer_part_only() -> None:
    """A prefix of length zero still returns the exact integer part."""
    result = radix_prefix(_value((1, 0, -2), 1), 10, 0)
    assert result.integer_part == 1
    assert result.fractional_digits == ()


def test_hexadecimal_prefix_uses_the_declared_base() -> None:
    """Golden ratio (1 + sqrt(5))/2 has base-16 prefix 1.9e3779b9..."""
    result = radix_prefix(_value((1, -1, -1), 1), 16, 8)
    assert result.integer_part == 1
    assert result.fractional_digits == (9, 14, 3, 7, 7, 9, 11, 9)


def test_non_golden_quadratic_uses_its_own_root_axis() -> None:
    """4x^2-4x-1 = 0 selects (1+sqrt(2))/2, whose base-16 prefix is 1.3504f3..."""
    result = radix_prefix(_value((4, -4, -1), 1), 16, 8)
    assert result.integer_part == 1
    assert result.fractional_digits == (3, 5, 0, 4, 15, 3, 3, 3)


def test_out_of_range_base_is_rejected() -> None:
    """Bases outside [2, 36] are outside the admitted alphabet."""
    with pytest.raises(OperationDomainValidationError):
        radix_prefix(_value((2, -1), 0), 1, 4)


def test_over_bound_places_is_a_resource_rejection() -> None:
    """A prefix longer than the coefficient envelope is refused, not truncated."""
    with pytest.raises(OperationResourceAdmissionError):
        radix_prefix(_value((1, 0, -2), 1), 10, MAX_RADIX_PLACES + 1)


def test_scaled_coefficient_growth_is_admitted_before_root_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transformed polynomial envelope is checked before SymPy isolation."""
    monkeypatch.setattr(radix_module, "MAX_RADIX_SCALED_COEFFICIENT_DIGITS", 2)
    with pytest.raises(OperationResourceAdmissionError, match="scaled defining"):
        radix_prefix(_value((1, 0, -2), 1), 10, 1)


def test_result_allocation_is_admitted_before_root_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retained source and digit list have an allocation envelope."""
    monkeypatch.setattr(radix_module, "MAX_RADIX_RESULT_ALLOCATION_UNITS", 1)
    with pytest.raises(OperationResourceAdmissionError, match="result allocation"):
        radix_prefix(_value((1, 0, -2), 1), 10, 1)


def test_prefix_round_trips_through_strict_json() -> None:
    """The declared prefix survives strict JSON serialization unchanged."""
    result = radix_prefix(_value((1, 0, -2), 1), 10, 12)
    restored = RadixPrefixResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json")), strict=True
    )
    assert restored == result


@pytest.mark.parametrize("polynomial", [(2, 0, -4), (1, 0, -1), (1, 0, 0, -2, 0)])
def test_noncanonical_algebraic_sources_are_rejected(
    polynomial: tuple[int, ...],
) -> None:
    with pytest.raises(OperationDomainValidationError):
        radix_prefix(
            RealAlgebraicValue(polynomial=polynomial, real_root_index=0), 10, 10
        )


def test_negative_boundary_root_uses_a_wider_integer_part_carrier() -> None:
    """floor of the negative root of x^2 + A x - A is -10**1000."""
    magnitude = 10**1000 - 1
    result = radix_prefix(_value((1, magnitude, -magnitude), 0), 10, 0)
    assert result.integer_part == -(10**1000)


def test_package_exports_the_native_radix_entrypoint() -> None:
    from jacobian.math.number_theory import algebraic_numbers

    assert algebraic_numbers.radix_prefix is radix_prefix
    assert "radix_prefix" in algebraic_numbers.__all__
