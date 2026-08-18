from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jacobian.math import arithmetic


@given(st.integers())
def test_absolute_value_and_sign_preserve_integer_invariants(value: int) -> None:
    assert arithmetic.absolute_value(value) >= 0
    assert arithmetic.absolute_value(value) * arithmetic.sign(value) == value


def test_exact_rational_operations() -> None:
    assert arithmetic.sum_rationals(Fraction(1, 3), Fraction(1, 6)) == Fraction(1, 2)
    assert arithmetic.quotient(Fraction(2, 3), 4) == Fraction(1, 6)
    assert arithmetic.reciprocal(Fraction(-2, 3)) == Fraction(-3, 2)


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
