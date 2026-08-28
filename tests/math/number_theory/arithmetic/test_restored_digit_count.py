from jacobian.math.number_theory.arithmetic._integers import INTEGER_OPERATIONS
from jacobian.math.number_theory.arithmetic._operations import decimal_digit_count
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def test_decimal_digit_count_handles_signed_canonical_integers() -> None:
    assert decimal_digit_count(IntegerValue(value="-12345")) == IntegerValue(value="5")


def test_decimal_digit_count_is_published_with_an_invocation_example() -> None:
    operation = next(
        operation
        for operation in INTEGER_OPERATIONS
        if operation.operation_id == "integer.compute.decimal_digit_count"
    )
    request = operation.request_type.model_validate(operation.examples[0].input)
    result = operation.run(request)

    assert result == IntegerValue(value="5")
