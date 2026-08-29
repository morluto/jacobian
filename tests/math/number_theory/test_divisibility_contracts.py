"""Public-contract regressions for divisibility-owned models."""

from typing import Any

import pytest

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.number_theory._divisibility import DIVISIBILITY_OPERATIONS
from jacobian.math.number_theory._divisibility_models import (
    DivisibilityRequest,
    ExtendedGcdResult,
    IntegerPairRequest,
    ValuationRequest,
)
from jacobian.math.number_theory.arithmetic.operations import (
    absolute_value,
    aliquot_sum,
    are_coprime,
    divides,
    divisor_count,
    divisor_sum,
    extended_gcd,
    integer_gcd,
    integer_lcm,
    is_abundant,
    is_deficient,
    is_even,
    is_odd,
    is_perfect,
    is_square,
    prime_valuation,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def _operation(operation_id: str) -> MathTool[Any, Any]:
    return next(
        operation
        for operation in DIVISIBILITY_OPERATIONS
        if operation.operation_id == operation_id
    )


@pytest.mark.parametrize(
    ("operation_id", "request_type", "result_type", "request_fields"),
    (
        (
            "integer.compute.extended_gcd",
            IntegerPairRequest,
            ExtendedGcdResult,
            {"left", "right"},
        ),
        (
            "integer.compute.valuation",
            ValuationRequest,
            IntegerValue,
            {"value", "prime"},
        ),
    ),
)
def test_divisibility_declarations_keep_their_owner_local_contracts(
    operation_id: str,
    request_type: type[IntegerPairRequest | DivisibilityRequest | ValuationRequest],
    result_type: type[IntegerValue | ExtendedGcdResult] | None,
    request_fields: set[str],
) -> None:
    operation = _operation(operation_id)

    assert operation.request_type is request_type
    assert set(request_type.model_json_schema()["properties"]) == request_fields
    if result_type is not None:
        assert operation.result_type is result_type


def test_divisibility_contracts_retain_their_typed_admission_errors() -> None:
    with pytest.raises(OperationDomainValidationError, match="divisor"):
        divides(0, 1)
    with pytest.raises(OperationDomainValidationError, match="nonzero"):
        prime_valuation(0, 2)
    with pytest.raises(OperationDomainValidationError, match="prime"):
        prime_valuation(1, 4)


def test_gcd_result_composes_with_arithmetic_integer_consumers() -> None:
    gcd = integer_gcd(-84, 30)

    assert type(gcd) is IntegerValue
    assert absolute_value(gcd) == IntegerValue(value="6")


def test_native_divisibility_vocabulary_remains_available() -> None:
    assert integer_lcm(12, 18) == IntegerValue(value="36")
    assert extended_gcd(84, 30).gcd == "6"
    assert divisor_count(36) == IntegerValue(value="9")
    assert divisor_sum(12) == IntegerValue(value="28")
    assert aliquot_sum(12) == IntegerValue(value="16")
    assert are_coprime(35, 12).holds
    assert divides(7, 42).holds
    assert is_even(42).holds
    assert is_odd(41).holds
    assert is_square(49).holds
    assert is_perfect(28).holds
    assert is_abundant(12).holds
    assert is_deficient(8).holds
