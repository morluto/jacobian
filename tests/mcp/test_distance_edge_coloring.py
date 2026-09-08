"""Distance palettes retain their exact graph source across live MCP."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.geometry.exact.distance_edge_coloring._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_distance_coloring_native_mcp_schema_and_graph_composition() -> None:
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
            parameters = await client.call_tool(
                "math.run",
                {
                    "operation_id": "hypergraph.parameters.compute",
                    "payload": {"hypergraph": output["coloring"]["hypergraph"]},
                },
            )
            assert not parameters.is_error
            assert parameters.structured_content is not None
            assert parameters.structured_content["output"]["edge_count"] == 6

    asyncio.run(scenario())
