from __future__ import annotations

import asyncio
import json

from jacobian.math.graphs.optimization._chromatic_bipartition import (
    CHROMATIC_BIPARTITION_OPERATION,
    ChromaticBipartitionRequest,
)
from jacobian.mcp.server import create_server
from mcp import Client


def test_live_mcp_split_round_trip_and_chromatic_consumers() -> None:
    async def scenario() -> None:
        payload = CHROMATIC_BIPARTITION_OPERATION.examples[0].input
        request = ChromaticBipartitionRequest.model_validate(payload)
        expected = CHROMATIC_BIPARTITION_OPERATION.run(request).model_dump(mode="json")
        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": CHROMATIC_BIPARTITION_OPERATION.operation_id,
                    "payload": payload,
                },
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            assert output == expected
            round_trip = json.loads(json.dumps(output))
            assert round_trip == output
            for side in (output["side_a"], output["side_b"]):
                induced_edges = [
                    edge
                    for edge in output["graph"]["edges"]
                    if edge[0] in side and edge[1] in side
                ]
                consumed = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "graph.invariant.chromatic_number.compute",
                        "payload": {
                            "graph": {"vertices": side, "edges": induced_edges}
                        },
                    },
                )
                assert not consumed.is_error
                assert consumed.structured_content is not None
                assert consumed.structured_content["output"]["status"] == "EXACT"

    asyncio.run(scenario())
