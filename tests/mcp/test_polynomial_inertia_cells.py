"""Complete exact inertia partition through native and live MCP boundaries."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.matrices.inertia_cells._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_inertia_cells_native_mcp_schema_parity() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        validate(payload, tool.request_type.model_json_schema())
        request = tool.request_type.model_validate_json(json.dumps(payload))
        native = tool.run(request).model_dump(mode="json")
        async with Client(create_server(), raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not result.is_error
            assert result.structured_content is not None
            output = result.structured_content["output"]
            assert output == native
            validate(output, tool.result_type.model_json_schema())
            left = output["cells"][2]["parameter"]["value"]
            right = output["cells"][4]["parameter"]["value"]
            comparison = await client.call_tool(
                "math.run",
                {
                    "operation_id": "algebraic_number.compare",
                    "payload": {"left": left, "right": right},
                },
            )
            assert not comparison.is_error
            assert comparison.structured_content is not None
            assert comparison.structured_content["output"]["order"] == "LT"

    asyncio.run(scenario())
