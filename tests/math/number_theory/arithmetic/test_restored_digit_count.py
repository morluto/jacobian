from jacobian.math.number_theory.arithmetic._integers import decimal_digit_count
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def test_decimal_digit_count_handles_signed_canonical_integers() -> None:
    assert decimal_digit_count(IntegerValue(value="-12345")) == IntegerValue(value="5")
