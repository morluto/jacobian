"""Large sparse division survives dispatch, serialization, and the MCP boundary."""

import asyncio
import json
from math import comb
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.multivariate._division import (
    MultivariateDivisionRequest,
    MultivariateDivisionResult,
)
from jacobian.math.polynomials.multivariate.operations import (
    multivariate_division,
    verify_multivariate_division,
)
from jacobian.mcp.server import create_server
from mcp import Client

_OPERATION = "polynomial.multivariate.divide.compute"
_AXES = ["u", "x", "z", "v", "w", "r", "y", "t"]


def _payload(degree: int, order: str = "lex") -> dict[str, Any]:
    def polynomial(terms: list[tuple[int, dict[str, int]]]) -> dict[str, Any]:
        encoded = [
            {
                "coefficient": {"num": str(coefficient), "den": "1"},
                "exponents": [powers.get(variable, 0) for variable in _AXES],
            }
            for coefficient, powers in terms
        ]
        return {
            "domain": "QQ",
            "variables": _AXES,
            "polynomial": {
                "terms": sorted(
                    encoded, key=lambda term: term["exponents"], reverse=True
                )
            },
        }

    return {
        "left": polynomial([(1, {"x": degree})]),
        "right": polynomial(
            [(1, {"x": 1}), (-1, {"y": 1}), (-1, {"z": 1}), (-1, {"w": 1})]
        ),
        "monomial_order": order,
    }


@pytest.mark.parametrize("degree,order", [(17, "lex"), (18, "grlex"), (28, "grevlex")])
def test_public_large_result_round_trip_and_real_consumer(
    degree: int, order: str
) -> None:
    catalog = Catalog.open()
    payload = _payload(degree, order)
    request = MultivariateDivisionRequest.model_validate_json(json.dumps(payload))
    native = multivariate_division(request.left, request.right, request.monomial_order)
    output = invoke_operation(_OPERATION, payload, catalog).output
    assert output == native.model_dump(mode="json")
    declaration = catalog.operation(_OPERATION)
    assert declaration is not None
    Draft202012Validator(
        declaration.result_type.model_json_schema(mode="serialization")
    ).validate(output)
    decoded = MultivariateDivisionResult.model_validate_json(json.dumps(output))
    assert verify_multivariate_division(decoded)
    assert (
        list(decoded.quotient.variables) == list(decoded.remainder.variables) == _AXES
    )
    assert len(decoded.quotient.polynomial.terms) == comb(degree + 2, 3)
    assert len(decoded.remainder.polynomial.terms) == comb(degree + 2, 2)
    assert (
        sum(t.coefficient.as_fraction() for t in decoded.quotient.polynomial.terms)
        == (3**degree - 1) // 2
    )
    assert (
        sum(t.coefficient.as_fraction() for t in decoded.remainder.polynomial.terms)
        == 3**degree
    )
    assert all(
        t.exponents[0] == t.exponents[3] == t.exponents[5] == t.exponents[7] == 0
        for p in (decoded.quotient, decoded.remainder)
        for t in p.polynomial.terms
    )
    false_source = decoded.model_copy(update={"left": decoded.right})
    assert not verify_multivariate_division(false_source)
    false_result = decoded.model_copy(update={"quotient": decoded.remainder})
    assert not verify_multivariate_division(false_result)


def test_mcp_refusal_and_recovery_keep_typed_support_diagnostic() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            refused = await client.call_tool(
                "math.run", {"operation_id": _OPERATION, "payload": _payload(29)}
            )
            assert refused.is_error
            content = refused.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert diagnostic["stage"] == "resource_admission"
            assert diagnostic["errors"][0]["code"] == "polynomial.division.support"
            response = await client.call_tool(
                "math.run", {"operation_id": _OPERATION, "payload": _payload(18)}
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            decoded = MultivariateDivisionResult.model_validate_json(json.dumps(output))
            assert len(decoded.quotient.polynomial.terms) == 1140
            assert len(decoded.remainder.polynomial.terms) == 190
            assert verify_multivariate_division(decoded)

    asyncio.run(scenario())
