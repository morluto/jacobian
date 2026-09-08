"""Native/MCP parity for exact rational covariant derivatives."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.geometry.differential.rational_tensor.covariant_derivative._tools import (
    TOOLS,
)
from jacobian.mcp.server import create_server
from mcp import Client


def test_live_rational_covariant_derivative_parity() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        validate(payload, tool.request_type.model_json_schema())
        request = tool.request_type.model_validate_json(json.dumps(payload))
        expected = tool.run(request).model_dump(mode="json")
        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            validate(output, tool.result_type.model_json_schema())
            assert output == expected
            assert output["variance"] == ["COVARIANT"]
            assert output["components"][0]["numerator"]["terms"]

    asyncio.run(scenario())
