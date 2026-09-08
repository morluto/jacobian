"""MCP parity for the edge-colored embedding operation."""

import asyncio
import json

from tests.mcp.test_direct_operations import _server

from jacobian.math.graphs.morphisms.edge_colored_pattern._tools import TOOLS
from mcp import Client


def test_colored_pattern_live_mcp_parity() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        request = tool.request_type.model_validate_json(json.dumps(payload))
        expected = tool.run(request)
        async with Client(_server(tool.operation_id), raise_exceptions=True) as client:
            response = await client.call_tool(tool.operation_id, payload)
            assert response.structured_content is not None
            assert (
                tool.result_type.model_validate_json(
                    json.dumps(response.structured_content)
                )
                == expected
            )

    asyncio.run(scenario())
