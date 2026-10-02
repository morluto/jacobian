"""Public GCD rejects unrepresentable witnesses without a host failure."""

import asyncio
import json

from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials._models import PolynomialGcdResult
from jacobian.mcp.server import create_server
from mcp import Client


def _monomial(degree: int) -> dict[str, object]:
    return {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [degree]}]
        },
    }


def test_bezout_public_height_boundary_and_recovery() -> None:
    operation = "polynomial.compute.gcd"
    right = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": str(-(10**255)), "den": "1"}, "exponents": [1]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
            ]
        },
    }
    accepted = {"left": _monomial(128), "right": right}
    refused = {"left": _monomial(129), "right": right}
    expected = invoke_operation(operation, accepted, Catalog.open()).output
    decoded = PolynomialGcdResult.model_validate_json(json.dumps(expected))
    assert decoded.bezout.left_multiplier.polynomial.terms[0].coefficient.num == 10 ** (
        255 * 128
    )
    assert len(decoded.bezout.right_multiplier.polynomial.terms) == 128

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            failure = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": refused}
            )
            assert failure.is_error
            content = failure.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert diagnostic["stage"] == "resource_admission"
            assert (
                diagnostic["errors"][0]["code"] == "polynomial.gcd.coefficient_height"
            )
            recovered = await client.call_tool(
                "math.run",
                {
                    "operation_id": operation,
                    "payload": {"left": _monomial(129), "right": _monomial(129)},
                },
            )
            assert not recovered.is_error
            assert recovered.structured_content is not None
            assert recovered.structured_content["output"]["gcd"] == _monomial(129)
            response = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": accepted}
            )
            assert not response.is_error
            assert response.structured_content is not None
            assert response.structured_content["output"] == expected

    asyncio.run(scenario())
