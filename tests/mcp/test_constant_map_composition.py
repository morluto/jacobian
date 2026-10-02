"""Constant rational-map admission and accepted wire-result composition."""

import asyncio
import json
from typing import Any

from mcp.types import TextContent

from jacobian.math.polynomials.rational_functions.composition import (
    RationalFunctionMapComposition,
)
from jacobian.mcp.server import create_server
from mcp import Client


def _function(
    variables: list[str], numerator: str, exponent: list[int]
) -> dict[str, Any]:
    return {
        "variables": variables,
        "numerator": {
            "terms": [
                {"coefficient": {"num": numerator, "den": "1"}, "exponents": exponent}
            ]
        },
        "denominator": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": [0] * len(variables),
                }
            ]
        },
    }


def test_constant_map_mcp_height_refusal_and_recovery() -> None:
    async def scenario() -> None:
        operation_id = "rational_function_map.compose.compute"
        inner = {
            "source_variables": [],
            "target_coordinates": ["u"],
            "components": [_function([], "1" + "0" * 127, [])],
        }
        outer = {
            "source_variables": ["u"],
            "target_coordinates": ["v"],
            "components": [_function(["u"], "1", [34])],
        }
        async with Client(create_server(), raise_exceptions=False) as client:
            for exponent in (34, 64):
                outer["components"] = [_function(["u"], "1", [exponent])]
                refused = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": operation_id,
                        "payload": {"outer": outer, "inner": inner},
                    },
                )
                assert refused.is_error
                assert isinstance(refused.content[0], TextContent)
                text = refused.content[0].text
                diagnostic = json.loads(text[text.index("{") :])
                assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
                assert (
                    diagnostic["errors"][0]["code"]
                    == "rational_function_map.compose.result_height"
                )
            outer["components"] = [_function(["u"], "1", [1])]
            accepted = await client.call_tool(
                "math.run",
                {
                    "operation_id": operation_id,
                    "payload": {"outer": outer, "inner": inner},
                },
            )
            assert not accepted.is_error
            assert accepted.structured_content is not None
            output = accepted.structured_content["output"]
            restored = RationalFunctionMapComposition.model_validate_json(
                json.dumps(output)
            )
            assert restored.composite.source_variables == ()
            assert (
                restored.composite.components[0].numerator.terms[0].coefficient.num
                == 10**127
            )
            outer["source_variables"] = ["v"]
            outer["components"] = [_function(["v"], "1", [1])]
            repeated = await client.call_tool(
                "math.run",
                {
                    "operation_id": operation_id,
                    "payload": {"outer": outer, "inner": output["composite"]},
                },
            )
            assert not repeated.is_error
            assert repeated.structured_content is not None
            assert (
                repeated.structured_content["output"]["composite"]
                == output["composite"]
            )

    asyncio.run(scenario())
