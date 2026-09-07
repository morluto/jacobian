"""Completion resource errors and successful values through MCP."""

import asyncio
import json

from mcp.types import TextContent

from jacobian.math.matrices.completion._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_resource_rejection_and_completion_recovery() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        request = tool.request_type.model_validate_json(json.dumps(payload))
        too_large = {
            "matrix": {
                "graph": {"vertex_count": 257, "edges": []},
                "specified_entries": [
                    {"row": i, "column": i, "value": {"num": "1", "den": "1"}}
                    for i in range(257)
                ],
            }
        }
        async with Client(create_server(), raise_exceptions=False) as client:
            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": too_large,
                },
            )
            assert rejected.is_error
            content = rejected.content[0]
            assert isinstance(content, TextContent)
            failure = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert failure["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert failure["stage"] == "resource_admission"
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": payload,
                },
            )
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["output"] == tool.run(request).model_dump(
                mode="json"
            )

    asyncio.run(scenario())
