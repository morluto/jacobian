"""One discoverable QQ evaluation contract across its exact execution regimes."""

import asyncio
import json
from fractions import Fraction
from typing import Any

from mcp.types import TextContent

from jacobian.math.polynomials.maps._models import EvalRequest, EvalResult
from jacobian.math.polynomials.maps.operations import evaluate_polynomial
from jacobian.mcp.server import create_server
from mcp import Client

_OPERATION = "polynomial.map.evaluate"
_RETIRED = "polynomial.rational.compute.evaluate"


def _payload(
    variables: list[str], terms: list[list[int]], coordinates: list[int]
) -> dict[str, Any]:
    return {
        "polynomial": {
            "domain": "QQ",
            "variables": variables,
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": "1", "den": "1"}, "exponents": key}
                    for key in terms
                ]
            },
        },
        "point": {
            "variables": variables,
            "values": [{"num": str(value), "den": "1"} for value in coordinates],
        },
    }


def test_mcp_discovers_one_qq_evaluator_with_both_visible_regimes() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            for arity in ("univariate", "bivariate"):
                found = await client.call_tool(
                    "math.find",
                    {
                        "query": f"evaluate a {arity} polynomial over QQ at a rational point"
                    },
                )
                assert not found.is_error
                assert found.structured_content is not None
                matches = found.structured_content["matches"]
                ids = [item["operation_id"] for item in matches]
                assert ids[0] == _OPERATION
                assert ids.count(_OPERATION) == 1
                assert _RETIRED not in ids
            inspected = await client.call_tool(
                "math.find", {"operation_id": _OPERATION}
            )
            assert inspected.structured_content is not None
            operation = inspected.structured_content["operation"]
            assert "degree 127" in operation["description"]
            assert "total degree 64" in operation["description"]
            for example in operation["examples"]:
                result = await client.call_tool(
                    "math.run",
                    {"operation_id": _OPERATION, "payload": example["input"]},
                )
                assert not result.is_error
            retired = await client.call_tool(
                "math.run", {"operation_id": _RETIRED, "payload": {}}
            )
            assert retired.is_error
            assert isinstance(retired.content[0], TextContent)
            assert "unknown operation" in retired.content[0].text

    asyncio.run(scenario())


def test_mcp_unified_evaluation_exact_degree_boundaries_zero_and_multivariate() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            cases = [
                (_payload([name], [[degree]], [-2]), Fraction((-2) ** degree))
                for name in ("x", "t")
                for degree in (64, 65, 127)
            ]
            cases.extend(
                [
                    (_payload(["x", "y"], [[1, 0], [0, 1]], [2, 2]), Fraction(4)),
                    (_payload(["t"], [], [2]), Fraction()),
                    (_payload(["y", "x"], [], [-2, 3]), Fraction()),
                ]
            )
            for payload, expected in cases:
                result = await client.call_tool(
                    "math.run", {"operation_id": _OPERATION, "payload": payload}
                )
                assert not result.is_error
                assert result.structured_content is not None
                decoded = EvalResult.model_validate_json(
                    json.dumps(result.structured_content["output"]), strict=True
                )
                assert decoded.value.as_fraction() == expected
                request = EvalRequest.model_validate_json(
                    json.dumps(payload), strict=True
                )
                assert decoded == evaluate_polynomial(request.polynomial, request.point)
            refused = await client.call_tool(
                "math.run",
                {"operation_id": _OPERATION, "payload": _payload(["x"], [[128]], [1])},
            )
            assert refused.is_error
            assert isinstance(refused.content[0], TextContent)
            diagnostic = json.loads(
                refused.content[0].text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "INVALID_REQUEST"
            assert "127-degree" in diagnostic["errors"][0]["message"]

    asyncio.run(scenario())
