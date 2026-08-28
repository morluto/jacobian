from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.arithmetic._operations import decimal_digit_count
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def test_decimal_digit_count_handles_signed_canonical_integers() -> None:
    assert decimal_digit_count(IntegerValue(value="-12345")) == IntegerValue(value="5")


def test_decimal_digit_count_is_published_with_an_invocation_example() -> None:
    result = invoke_operation(
        "integer.compute.decimal_digit_count", {"value": "12345"}, Catalog.open()
    )

    assert result.output == {"value": "5"}
