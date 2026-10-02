"""The exact repair value is the success contract at the real MCP boundary."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.graphs.coloring._models import (
    PrecoloringEdgeRepairRequest,
    PrecoloringEdgeRepairResult,
)
from jacobian.math.graphs.coloring._tools import compute_precoloring_edge_repair
from jacobian.mcp.server import create_server
from mcp import Client


def test_repair_inspection_and_execution_publish_the_value_directly() -> None:
    async def scenario() -> None:
        operation_id = "graph.coloring.precoloring_edge_repair.compute"
        async with Client(create_server(), raise_exceptions=False) as client:
            inspected = await client.call_tool(
                "math.find", {"operation_id": operation_id}
            )
            assert not inspected.is_error
            assert inspected.structured_content is not None
            contract = inspected.structured_content["operation"]
            schema = contract["output_schema"]
            assert "status" not in schema["properties"]
            assert "status" not in schema.get("required", ())
            for payload in (
                {"graph": {"vertex_count": 3, "edges": []}, "colors": 3},
                contract["examples"][0]["input"],
            ):
                request = PrecoloringEdgeRepairRequest.model_validate_json(
                    json.dumps(payload)
                )
                native = compute_precoloring_edge_repair(request)
                reply = await client.call_tool(
                    "math.run", {"operation_id": operation_id, "payload": payload}
                )
                assert not reply.is_error
                assert reply.structured_content is not None
                output = reply.structured_content["output"]
                assert "status" not in output
                assert output == native.model_dump(mode="json")
                validate(output, schema)
                decoded = PrecoloringEdgeRepairResult.model_validate_json(
                    json.dumps(output)
                )
                assert decoded == native
                coloring = decoded.coloring.coloring
                repaired = tuple(
                    i
                    for i, (u, v) in enumerate(decoded.graph.edges)
                    if coloring[u] == coloring[v]
                )
                assert decoded.repaired_edge_indices == repaired
                assert decoded.repaired_edge_count == len(repaired)

    asyncio.run(scenario())
