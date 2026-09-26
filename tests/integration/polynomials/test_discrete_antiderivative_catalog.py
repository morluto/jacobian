"""Catalog invocation contract for rational discrete antiderivatives."""

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials._discrete_antiderivative import (
    RationalDiscreteAntiderivativeResult,
)
from jacobian.math.polynomials._discrete_antiderivative_tools import (
    RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION,
)
from jacobian.math.polynomials.values import RationalPolynomial


def test_catalog_invocation_returns_the_declared_typed_result() -> None:
    catalog = Catalog.open()
    invocation = RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.examples[0]
    result = invoke_operation(
        RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.operation_id,
        invocation.input,
        catalog,
    )
    assert set(result.output) == {
        "source",
        "variable",
        "antiderivative",
        "reconstructed_difference",
    }
    typed_result = RationalDiscreteAntiderivativeResult.model_validate_json(
        json.dumps(result.output)
    )
    expected_source = RationalPolynomial.model_validate_json(
        json.dumps(invocation.input["polynomial"])
    )
    expected_antiderivative = RationalPolynomial.model_validate(
        {
            "variables": ["k"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": 1, "den": 3}, "exponents": [3]},
                    {"coefficient": {"num": -1, "den": 2}, "exponents": [2]},
                    {"coefficient": {"num": 1, "den": 6}, "exponents": [1]},
                ]
            },
        }
    )
    assert typed_result.source == expected_source
    assert typed_result.variable == "k"
    assert typed_result.antiderivative == expected_antiderivative
    assert typed_result.reconstructed_difference == expected_source
