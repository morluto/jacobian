from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jacobian.math.number_theory import arithmetic


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

    with pytest.raises(ZeroDivisionError, match=r"zero|division by zero"):
        operation(zero)  # type: ignore[operator]
    with pytest.raises(ZeroDivisionError, match=r"zero|division by zero"):
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
    with pytest.raises(ValueError, match="nonzero"):
        arithmetic.primitive_integer_vector((Fraction(0), Fraction(0)))


@pytest.mark.parametrize(
    "operation", [arithmetic.reciprocal, lambda x: arithmetic.quotient(1, x)]
)
def test_zero_division_is_explicit(operation: object) -> None:
    with pytest.raises(ZeroDivisionError, match=r"zero|division by zero"):
        operation(0)  # type: ignore[operator]


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the arithmetic public API."""
    expected = (
        "IntegerValue",
        "absolute_value",
        "integerize_rational_vector",
        "primitive_integer_vector",
        "quotient",
        "reciprocal",
        "sign",
        "sum_rationals",
    )
    assert tuple(arithmetic.__all__) == expected
    assert len(arithmetic.__all__) == len(set(arithmetic.__all__))
    assert all(not name.startswith("_") for name in arithmetic.__all__)
    assert all(hasattr(arithmetic, name) for name in arithmetic.__all__)
