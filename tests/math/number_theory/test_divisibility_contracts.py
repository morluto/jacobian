"""Public-contract regressions for divisibility-owned models."""

from typing import Any

import pytest
from tests.math.number_theory._validation import expect_validation

from jacobian.catalog.models import MathTool
from jacobian.math.arithmetic.values import IntegerValue
from jacobian.math.number_theory._divisibility import DIVISIBILITY_OPERATIONS
from jacobian.math.number_theory._divisibility_models import (
    DivisibilityRequest,
    ExtendedGcdResult,
    IntegerPairRequest,
    ValuationRequest,
)


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
            "integer.compute.gcd",
            IntegerPairRequest,
            IntegerValue,
            {"left", "right"},
        ),
        (
            "integer.compute.lcm",
            IntegerPairRequest,
            IntegerValue,
            {"left", "right"},
        ),
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
        ("integer.decide.divides", DivisibilityRequest, None, {"divisor", "dividend"}),
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
    with expect_validation("number_theory.divisor_must_be_nonzero"):
        DivisibilityRequest(divisor="0", dividend="1")
    with expect_validation("number_theory.valuation_requires_nonzero_value"):
        ValuationRequest(value="0", prime="2")
    with expect_validation("number_theory.valuation_requires_a_prime_absolute_base_2"):
        ValuationRequest(value="1", prime="4")
