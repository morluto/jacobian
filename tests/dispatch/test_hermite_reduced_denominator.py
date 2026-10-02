"""Reduced primitive admission is shared by public rational-function callers."""

import json

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.rational_functions._models import HermiteReductionResult
from jacobian.math.polynomials.rational_functions.operations import (
    verify_hermite_reduction,
)


@pytest.mark.parametrize(
    "operation_id",
    [
        "rational_function.hermite_reduction.compute",
        "rational_function.rational_primitive.compute",
        "rational_function.formal_antiderivative.compute",
        "rational_function.logarithmic_differential.compute",
        "rational_function.partial_fractions.compute",
    ],
)
def test_public_hermite_callers_admit_cancellable_denominator(
    operation_id: str,
) -> None:
    denominator = "9" * 128
    payload = {
        "function": {
            "variables": ["x"],
            "numerator": {
                "terms": [
                    {"coefficient": {"num": "2", "den": denominator}, "exponents": [1]}
                ]
            },
            "denominator": {
                "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
            },
        }
    }
    catalog = Catalog.open()
    result = invoke_operation(operation_id, payload, catalog)
    declaration = catalog.operation(operation_id)
    assert declaration is not None
    decoded = declaration.result_type.model_validate_json(json.dumps(result.output))
    assert decoded.model_dump(mode="json") == result.output
    if operation_id == "rational_function.hermite_reduction.compute":
        claim = HermiteReductionResult.model_validate_json(json.dumps(result.output))
        assert verify_hermite_reduction(claim)
        assert result.output["rational_part"]["numerator"]["terms"] == [
            {"coefficient": {"num": "1", "den": denominator}, "exponents": [2]}
        ]


@pytest.mark.parametrize(
    "operation_id",
    (
        "rational_function.hermite_reduction.compute",
        "rational_function.rational_primitive.compute",
        "rational_function.formal_antiderivative.compute",
    ),
)
def test_public_hermite_keeps_uncancelled_overflow_typed(operation_id: str) -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        invoke_operation(
            operation_id,
            {
                "function": {
                    "variables": ["x"],
                    "numerator": {
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "9" * 128},
                                "exponents": [1],
                            }
                        ]
                    },
                    "denominator": {
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [0],
                            }
                        ]
                    },
                }
            },
            Catalog.open(),
        )
    assert error.value.errors()[0]["type"] == "polynomial.hermite_reduction_budget"
