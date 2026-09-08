"""Rational map differential values survive the live MCP boundary unchanged."""

import asyncio
import json

from jacobian.math.polynomials.rational_functions.maps._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_native_mcp_and_derivative_component_composition() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        request = tool.request_type.model_validate_json(json.dumps(payload))
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not result.is_error
            assert result.structured_content is not None
            output = result.structured_content["output"]
            assert output == tool.run(request).model_dump(mode="json")
            composed = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function.gradient.compute",
                    "payload": {"function": output["entries"][0][0]},
                },
            )
            assert not composed.is_error

    asyncio.run(scenario())
