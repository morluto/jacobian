"""Exact decomposition through the live MCP contract."""

import asyncio
import json

from jacobian.math.matrices.chordal_psd._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_native_mcp_parity_and_non_psd_recovery() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        request = tool.request_type.model_validate_json(json.dumps(payload))
        async with Client(create_server(), raise_exceptions=False) as client:
            bad = dict(payload)
            bad["matrix"] = {
                "entries": [
                    [
                        {"num": "-1", "den": "1"}
                        if i == j
                        else {"num": "0", "den": "1"}
                        for j in range(3)
                    ]
                    for i in range(3)
                ]
            }
            rejected = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": bad}
            )
            assert rejected.is_error
            result = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["output"] == tool.run(request).model_dump(
                mode="json"
            )

            local = result.structured_content["output"]["terms"][0]["matrix"]
            inertia = await client.call_tool(
                "math.run",
                {
                    "operation_id": "matrix.inertia.compute",
                    "payload": {"matrix": local},
                },
            )
            assert not inertia.is_error
            assert inertia.structured_content is not None
            assert inertia.structured_content["output"]["n_positive"] == 1
            assert inertia.structured_content["output"]["n_negative"] == 0

    asyncio.run(scenario())
