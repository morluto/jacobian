"""Source retention and native equivalence through the real MCP boundary."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.graphs.bipartite._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_live_dulmage_mendelsohn_and_source_round_trip() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        validate(payload, tool.request_type.model_json_schema())
        request = tool.request_type.model_validate_json(json.dumps(payload))
        async with Client(create_server(), raise_exceptions=False) as client:
            invalid = json.loads(json.dumps(payload))
            invalid["graph"]["right_vertices"] = [1, 3]
            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": invalid,
                },
            )
            assert rejected.is_error
            result = await client.call_tool(
                "math.run", {"operation_id": tool.operation_id, "payload": payload}
            )
            assert not result.is_error
            assert result.structured_content is not None
            output = result.structured_content["output"]
            validate(output, tool.result_type.model_json_schema())
            assert output == tool.run(request).model_dump(mode="json")
            reused = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": {"graph": output["graph"]},
                },
            )
            assert not reused.is_error
            assert reused.structured_content is not None
            assert reused.structured_content["output"] == output
            matching = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.invariant.maximum_matching.compute",
                    "payload": {
                        "graph": {
                            "vertices": ["l0", "l1", "r0", "r1"],
                            "edges": [["l0", "r0"], ["l0", "r1"], ["l1", "r1"]],
                        }
                    },
                },
            )
            assert not matching.is_error
            assert matching.structured_content is not None
            assert (
                matching.structured_content["output"]["maximum_matching_cardinality"]
                == output["structural_rank"]
            )

    asyncio.run(scenario())
