"""Live exact Lebesgue profile parity and retained basis composition."""

import asyncio
import json

from jacobian.math.analysis.approximation.lebesgue._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_lebesgue_profile_native_mcp_and_basis_composition() -> None:
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
            basis = await client.call_tool(
                "math.run",
                {
                    "operation_id": "approximation.lagrange.basis.compute",
                    "payload": {"nodes": output["source"]["nodes"]},
                },
            )
            assert not basis.is_error
            assert basis.structured_content is not None
            assert basis.structured_content["output"] == output["basis"]

    asyncio.run(scenario())
