"""Exact unified polynomial evaluation and unchanged canonical composition."""

import json
from fractions import Fraction
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.polynomials.maps._models import EvalRequest, EvalResult
from jacobian.math.polynomials.maps.operations import evaluate_polynomial


def _payload(degree: int, name: str = "t") -> dict[str, Any]:
    return {
        "polynomial": {
            "variables": [name],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": "-3", "den": "7"}, "exponents": [degree]}
                ]
            },
        },
        "point": {"variables": [name], "values": [{"num": "2", "den": "3"}]},
    }


@pytest.mark.parametrize("degree", (64, 65, 127))
def test_unified_public_evaluation_is_exact_and_round_trips(degree: int) -> None:
    payload = _payload(degree)
    request = EvalRequest.model_validate_json(json.dumps(payload), strict=True)
    output = invoke_operation("polynomial.map.evaluate", payload, Catalog.open()).output
    result = EvalResult.model_validate_json(json.dumps(output), strict=True)
    assert result == evaluate_polynomial(request.polynomial, request.point)
    assert result.value.as_fraction() == Fraction(-3, 7) * Fraction(2, 3) ** degree
    # The returned canonical rational goes unchanged into a real scalar consumer.
    consumed = invoke_operation(
        "rational.compute.sum",
        {"left": output["value"], "right": {"num": "0", "den": "1"}},
        Catalog.open(),
    ).output
    assert consumed["value"] == output["value"]


def test_unified_evaluation_consumes_an_unchanged_degree_65_producer_result() -> None:
    payload = _payload(66)
    derivative = invoke_operation(
        "polynomial.rational.compute.derivative",
        {"polynomial": payload["polynomial"]},
        Catalog.open(),
    ).output
    payload["polynomial"] = derivative["derivative"]
    result = invoke_operation("polynomial.map.evaluate", payload, Catalog.open()).output
    decoded = EvalResult.model_validate_json(json.dumps(result), strict=True)
    assert decoded.value.as_fraction() == 66 * Fraction(-3, 7) * Fraction(2, 3) ** 65


def test_unified_evaluation_rejects_next_degree_and_implicit_point_coercion() -> None:
    with pytest.raises(OperationDomainValidationError):
        invoke_operation("polynomial.map.evaluate", _payload(128), Catalog.open())
    payload = _payload(65)
    payload["point"] = {"num": "2", "den": "3"}
    with pytest.raises(OperationRequestValidationError):
        invoke_operation("polynomial.map.evaluate", payload, Catalog.open())
    payload = _payload(65)
    payload["point"]["variables"] = ["x"]
    with pytest.raises(OperationDomainValidationError, match="complete ordered axis"):
        invoke_operation("polynomial.map.evaluate", payload, Catalog.open())
