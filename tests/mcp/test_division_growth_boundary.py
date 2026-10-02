"""Canonical division overflow is a public resource refusal with recovery."""

import asyncio
import json
from math import isqrt, prod
from typing import Any

from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.multivariate._division import MultivariateDivisionResult
from jacobian.math.polynomials.multivariate.operations import (
    verify_multivariate_division,
)
from jacobian.mcp.server import create_server
from mcp import Client


def _payload(degree: int) -> tuple[dict[str, Any], tuple[int, ...]]:
    primes = [p for p in range(7, 500) if all(p % d for d in range(2, isqrt(p) + 1))][
        : degree + 1
    ]
    denominators = []
    for prime in primes:
        value = prime
        while value * prime < 10**256:
            value *= prime
        denominators.append(value)
    left = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": str(denominators[i])},
                    "exponents": [i],
                }
                for i in range(degree, -1, -1)
            ]
        },
    }
    right = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                {"coefficient": {"num": "-1", "den": str(10**255)}, "exponents": [0]},
            ]
        },
    }
    return {"left": left, "right": right, "monomial_order": "lex"}, tuple(denominators)


def test_public_division_admission_boundary_and_server_recovery() -> None:
    operation = "polynomial.multivariate.divide.compute"
    admitted, denominators = _payload(63)
    rejected, _ = _payload(64)
    output = invoke_operation(operation, admitted, Catalog.open()).output
    decoded = MultivariateDivisionResult.model_validate_json(json.dumps(output))
    assert decoded.remainder.polynomial.terms[0].coefficient.den == 10 ** (
        255 * 63
    ) * prod(denominators)
    assert verify_multivariate_division(decoded)

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            refused = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": rejected}
            )
            assert refused.is_error
            content = refused.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert diagnostic["stage"] == "resource_admission"
            assert (
                diagnostic["errors"][0]["code"]
                == "polynomial.division.coefficient_height"
            )
            self_division = {**rejected, "right": rejected["left"]}
            recovered = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": self_division}
            )
            assert not recovered.is_error
            assert recovered.structured_content is not None
            self_result = MultivariateDivisionResult.model_validate_json(
                json.dumps(recovered.structured_content["output"])
            )
            assert self_result.quotient.polynomial.terms[
                0
            ].coefficient.as_integer_ratio() == (1, 1)
            assert not self_result.remainder.polynomial.terms
            response = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": admitted}
            )
            assert not response.is_error
            assert response.structured_content is not None
            assert response.structured_content["output"] == output

    asyncio.run(scenario())
