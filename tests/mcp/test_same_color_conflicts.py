"""Exact geometry-to-conflict-to-independence composition through live MCP."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.combinatorics.finite_structures.hypergraphs.same_color_conflicts._tools import (
    TOOLS,
)
from jacobian.math.geometry.exact.distance_edge_coloring._tools import (
    TOOLS as DISTANCE_TOOLS,
)
from jacobian.mcp.server import create_server
from mcp import Client


def test_conflicts_native_schema_and_complete_distance_chain() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        validate(payload, tool.request_type.model_json_schema())
        native = tool.run(tool.request_type.model_validate_json(json.dumps(payload)))
        async with Client(create_server(), raise_exceptions=False) as client:
            invalid = json.loads(json.dumps(payload))
            invalid["coloring"]["assignments"].pop()
            rejected = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": invalid}
            )
            assert rejected.is_error
            response = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            validate(output, tool.result_type.model_json_schema())
            assert output == native.model_dump(mode="json")
            distance_tool = DISTANCE_TOOLS[0]
            distances = await client.call_tool(
                "math.run",
                {
                    "operation_id": distance_tool.operation_id,
                    "payload": distance_tool.examples[0].input,
                },
            )
            assert not distances.is_error
            assert distances.structured_content is not None
            geometry = distances.structured_content["output"]
            conflicts = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": {"coloring": geometry["coloring"]},
                },
            )
            assert not conflicts.is_error
            assert conflicts.structured_content is not None
            conflict_output = conflicts.structured_content["output"]
            assert conflict_output["coloring"] == geometry["coloring"]
            assert len(conflict_output["provenance"]) == 7
            assert len(conflict_output["hypergraph"]["edges"]) == 5
            independent = await client.call_tool(
                "math.run",
                {
                    "operation_id": "hypergraph.independence_number.compute",
                    "payload": {"hypergraph": conflict_output["hypergraph"]},
                },
            )
            assert not independent.is_error
            assert independent.structured_content is not None
            assert independent.structured_content["output"]["status"] == "EXACT"
            assert independent.structured_content["output"]["independence_number"] == 2

    asyncio.run(scenario())
