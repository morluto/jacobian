"""Composition tests for the canonical exact integer value."""

from jacobian.math.arithmetic._operations import absolute_value, decimal_digit_sum
from jacobian.math.arithmetic.values import IntegerValue
from jacobian.math.number_theory._divisibility_operations import decide_even


def test_integer_operations_share_one_typed_canonical_value() -> None:
    source = IntegerValue(value="-42")

    absolute = absolute_value(source)
    digit_sum = decimal_digit_sum(absolute)

    assert type(absolute) is IntegerValue
    assert type(digit_sum) is IntegerValue
    assert absolute.value == "42"
    assert digit_sum.value == "6"
    assert decide_even(absolute).holds


def test_unary_integer_value_is_not_limited_by_primality_admission() -> None:
    value = IntegerValue(value="1" + "0" * 256)

    assert decimal_digit_sum(value) == IntegerValue(value="1")
