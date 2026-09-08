"""Native/MCP parity, authored-claim rejection, and recovery."""

import asyncio
import json

from jacobian.math.geometry.finite.cosets._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_coset_profile_mcp_recovery_and_native_parity() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        request = tool.request_type.model_validate_json(json.dumps(payload))
        invalid = request.model_dump(mode="json")
        invalid["subspace"]["basis"] = [[0, 0, 0]]
        async with Client(create_server(), raise_exceptions=False) as client:
            failure = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": invalid}
            )
            assert failure.is_error
            response = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not response.is_error
            assert response.structured_content is not None
            assert response.structured_content["output"] == tool.run(
                request
            ).model_dump(mode="json")

    asyncio.run(scenario())
