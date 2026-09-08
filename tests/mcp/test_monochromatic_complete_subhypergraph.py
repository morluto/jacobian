"""MCP projection and composition for monochromatic target profiles."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.combinatorics.finite_structures.hypergraphs.monochromatic_complete_subhypergraph._tools import (
    TOOLS,
)
from jacobian.mcp.server import create_server
from mcp import Client


def test_monochromatic_profile_native_mcp_schema_and_hypergraph_composition() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        validate(payload, tool.request_type.model_json_schema())
        native = tool.run(tool.request_type.model_validate_json(json.dumps(payload)))

        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            validate(output, tool.result_type.model_json_schema())
            assert output == native.model_dump(mode="json")

            parameters = await client.call_tool(
                "math.run",
                {
                    "operation_id": "hypergraph.parameters.compute",
                    "payload": {"hypergraph": output["hypergraph"]},
                },
            )
            assert not parameters.is_error
            assert parameters.structured_content is not None
            assert parameters.structured_content["output"]["uniform_size"] == 4

    asyncio.run(scenario())
