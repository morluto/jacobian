from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory import arithmetic
from jacobian.math.number_theory.arithmetic import operations as arithmetic_operations
from jacobian.math.number_theory.arithmetic._rational_models import RationalPairRequest
from jacobian.math.number_theory.arithmetic._rationals import (
    RATIONAL_OPERATIONS,
    product,
)


@given(st.integers())
def test_absolute_value_and_sign_preserve_integer_invariants(value: int) -> None:
    absolute = arithmetic.absolute_value(value)

    assert type(absolute) is arithmetic.IntegerValue
    assert absolute == arithmetic.IntegerValue(value=str(abs(value)))
    assert arithmetic.sign(value) * abs(value) == value
    assert arithmetic.sign(absolute) == arithmetic.sign(abs(value))


@given(st.integers())
def test_absolute_value_and_sign_compose_through_canonical_integer_value(
    value: int,
) -> None:
    absolute = arithmetic.absolute_value(arithmetic.IntegerValue(value=str(value)))

    assert type(absolute) is arithmetic.IntegerValue
    assert absolute == arithmetic.absolute_value(value)
    assert arithmetic.sign(absolute) == arithmetic.sign(abs(value))


def test_exact_rational_operations() -> None:
    assert arithmetic.sum_rationals(Fraction(1, 3), Fraction(1, 6)) == Fraction(1, 2)
    assert arithmetic.quotient(Fraction(2, 3), 4) == Fraction(1, 6)
    assert arithmetic.reciprocal(Fraction(-2, 3)) == Fraction(-3, 2)


def test_reciprocal_composes_with_the_absolute_value_producer() -> None:
    """Producer-consumer regression: the canonical integer value must compose."""

    absolute = arithmetic.absolute_value(-2)

    assert type(absolute) is arithmetic.IntegerValue
    assert arithmetic.reciprocal(absolute) == Fraction(1, 2)


def test_producer_chains_compose_through_the_canonical_integer_value() -> None:
    magnitude = arithmetic.absolute_value(-6)

    assert arithmetic.sum_rationals(magnitude, Fraction(1, 2)) == Fraction(13, 2)
    assert arithmetic.quotient(magnitude, 4) == Fraction(3, 2)
    assert arithmetic.integerize_rational_vector((magnitude, Fraction(1, 2))) == (12, 1)
    assert arithmetic.primitive_integer_vector((magnitude, Fraction(1, 2))) == (12, 1)


@given(st.integers(), st.integers().filter(lambda value: value != 0))
def test_rational_consumers_identify_plain_and_canonical_integers(
    left: int, right: int
) -> None:
    canonical_left = arithmetic.IntegerValue(value=str(left))
    canonical_right = arithmetic.IntegerValue(value=str(right))

    assert (
        arithmetic.sum_rationals(canonical_left, canonical_right)
        == arithmetic.sum_rationals(left, right)
        == Fraction(left) + Fraction(right)
    )
    assert arithmetic.quotient(canonical_left, canonical_right) == arithmetic.quotient(
        left, right
    )
    assert arithmetic.reciprocal(canonical_right) == arithmetic.reciprocal(right)
    assert arithmetic.integerize_rational_vector(
        (canonical_left, canonical_right)
    ) == arithmetic.integerize_rational_vector((left, right))
    assert arithmetic.primitive_integer_vector(
        (canonical_left, canonical_right)
    ) == arithmetic.primitive_integer_vector((left, right))


@pytest.mark.parametrize("operation", [arithmetic.reciprocal])
def test_zero_canonical_integer_rejection_matches_plain_zero(operation: object) -> None:
    zero = arithmetic.IntegerValue(value="0")

    with pytest.raises(OperationDomainValidationError):
        operation(zero)  # type: ignore[operator]
    with pytest.raises(OperationDomainValidationError):
        arithmetic.quotient(3, zero)


def test_integerize_and_normalize_exact_rational_vectors() -> None:
    values = (Fraction(-1, 2), Fraction(3, 4), Fraction(-1, 8))

    assert arithmetic.integerize_rational_vector(values) == (-4, 6, -1)
    assert arithmetic.primitive_integer_vector(values) == (4, -6, 1)


def test_integerize_rational_vector_uses_one_lcm_for_mixed_denominators() -> None:
    values = (Fraction(1, 6), Fraction(1, 4), Fraction(1, 9), Fraction(5, 18))

    assert arithmetic.integerize_rational_vector(values) == (6, 9, 4, 10)
    assert arithmetic.primitive_integer_vector(values) == (6, 9, 4, 10)


def test_primitive_integer_vector_rejects_zero_vector() -> None:
    with pytest.raises(OperationDomainValidationError, match="nonzero"):
        arithmetic.primitive_integer_vector((Fraction(0), Fraction(0)))


def test_native_integer_admission_uses_typed_domain_errors() -> None:
    value = arithmetic.IntegerValue(value="8")

    with pytest.raises(OperationDomainValidationError, match="base must be"):
        arithmetic_operations.base_digits(value, 1)
    with pytest.raises(OperationDomainValidationError, match="degree must be"):
        arithmetic_operations.nth_root(value, 0)


@pytest.mark.parametrize(
    "operation", [arithmetic.reciprocal, lambda x: arithmetic.quotient(1, x)]
)
def test_zero_division_is_explicit(operation: object) -> None:
    with pytest.raises(OperationDomainValidationError):
        operation(0)  # type: ignore[operator]


def test_complete_native_rational_arithmetic() -> None:
    left = Fraction(-7, 3)
    right = Fraction(2, 5)

    assert arithmetic.negate_rational(left) == Fraction(7, 3)
    assert arithmetic.rational_absolute_value(left) == Fraction(7, 3)
    assert arithmetic.difference_rationals(left, right) == Fraction(-41, 15)
    assert arithmetic.product_rationals(left, right) == Fraction(-14, 15)
    assert arithmetic.minimum_rational(left, right) == left
    assert arithmetic.maximum_rational(left, right) == right
    assert arithmetic.floor_rational(left) == -3
    assert arithmetic.ceiling_rational(left) == -2
    assert arithmetic.continued_fraction(left) == (-3, 1, 2)
    assert arithmetic.equal_rationals(left, Fraction(-7, 3)) is True
    assert arithmetic.less_than_rationals(left, right) is True


def test_published_product_rejects_unrepresentable_exact_result() -> None:
    left_denominator = "1" + "0" * (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    right_denominator = "9" * MAX_CANONICAL_RATIONAL_DIGITS
    request = RationalPairRequest(
        left=CanonicalRational(num="1", den=left_denominator),
        right=CanonicalRational(num="1", den=right_denominator),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        product(request)

    assert exc_info.value.errors()[0]["type"] == (
        "arithmetic.rational_result_exceeds_component_bound"
    )


def test_all_fourteen_rational_operations_are_published() -> None:
    assert len(RATIONAL_OPERATIONS) == 14
    assert len({operation.operation_id for operation in RATIONAL_OPERATIONS}) == 14


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the arithmetic public API."""
    expected = (
        "IntegerValue",
        "absolute_value",
        "aliquot_sum",
        "are_coprime",
        "ceiling_rational",
        "continued_fraction",
        "difference_rationals",
        "divides",
        "divisor_count",
        "divisor_sum",
        "equal_rationals",
        "extended_gcd",
        "floor_rational",
        "integer_gcd",
        "integer_lcm",
        "integerize_rational_vector",
        "is_abundant",
        "is_deficient",
        "is_even",
        "is_odd",
        "is_perfect",
        "is_square",
        "less_than_rationals",
        "maximum_rational",
        "minimum_rational",
        "negate_rational",
        "prime_valuation",
        "primitive_integer_vector",
        "product_rationals",
        "quotient",
        "rational_absolute_value",
        "reciprocal",
        "sign",
        "sum_rationals",
    )
    assert tuple(arithmetic.__all__) == expected
    assert len(arithmetic.__all__) == len(set(arithmetic.__all__))
    assert all(not name.startswith("_") for name in arithmetic.__all__)
    assert all(hasattr(arithmetic, name) for name in arithmetic.__all__)
