"""Live scalar-gradient composition and recovery."""

import asyncio
import json

from jacobian.math.polynomials.rational_functions.gradient._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_gradient_native_mcp_and_serialized_component_composition() -> None:
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
            repeated = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": {"function": output["partial_derivatives"][0]},
                },
            )
            assert not repeated.is_error

    asyncio.run(scenario())
