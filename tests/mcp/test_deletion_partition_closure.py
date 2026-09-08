"""Real MCP parity, schema validation, rejection recovery and graph reuse."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.graphs.coloring.deletion_partition_closure._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_live_closure() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payload = tool.examples[0].input
        validate(payload, tool.request_type.model_json_schema())
        request = tool.request_type.model_validate_json(json.dumps(payload))
        async with Client(create_server(), raise_exceptions=False) as client:
            invalid = json.loads(json.dumps(payload))
            invalid["template"]["rows"].pop()
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
            assert output == tool.run(request).model_dump(mode="json")
            reused = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": {"template": output["template"]},
                },
            )
            assert not reused.is_error
            assert reused.structured_content is not None
            assert reused.structured_content["output"] == output
            matching = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.invariant.maximum_matching.compute",
                    "payload": {"graph": output["graph"]},
                },
            )
            assert not matching.is_error
            assert matching.structured_content is not None
            assert (
                matching.structured_content["output"]["maximum_matching_cardinality"]
                == 0
            )

    asyncio.run(scenario())
