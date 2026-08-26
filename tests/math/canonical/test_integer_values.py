"""Composition tests for the canonical exact integer value."""

from jacobian.math import arithmetic
from jacobian.math.arithmetic._operations import decimal_digit_sum
from jacobian.math.arithmetic.values import IntegerValue
from jacobian.math.number_theory._divisibility_operations import decide_even


def test_public_native_functions_compose_through_the_shared_integer_value() -> None:
    source = IntegerValue(value="-42")

    absolute = arithmetic.absolute_value(source)

    assert type(absolute) is IntegerValue
    assert absolute == IntegerValue(value="42")
    assert arithmetic.sign(source) == -1
    assert arithmetic.sign(absolute) == 1
    assert decide_even(absolute).holds


def test_absolute_value_accepts_plain_python_integers() -> None:
    assert arithmetic.absolute_value(-42) == IntegerValue(value="42")
    assert arithmetic.sign(42) == 1


def test_unary_integer_value_is_not_limited_by_primality_admission() -> None:
    value = IntegerValue(value="1" + "0" * 256)

    assert decimal_digit_sum(value) == IntegerValue(value="1")
