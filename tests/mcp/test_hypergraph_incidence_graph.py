"""Live MCP composition for canonical hypergraph incidence graphs."""

import asyncio
import json

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    IncidenceGraphRequest,
    IncidenceGraphResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    incidence_graph,
)
from jacobian.mcp.server import create_server
from mcp import Client


def test_incidence_graph_round_trips_into_cycle_profile() -> None:
    async def scenario() -> None:
        payload = {
            "hypergraph": {
                "vertices": ["same", "isolated"],
                "edges": [["same", ["same"]], ["other", ["same"]]],
            }
        }
        request = IncidenceGraphRequest.model_validate_json(json.dumps(payload))
        expected = incidence_graph(request.hypergraph).model_dump(mode="json")
        async with Client(create_server(), raise_exceptions=False) as client:
            response = await client.call_tool(
                "math.run",
                {
                    "operation_id": "hypergraph.incidence_graph.compute",
                    "payload": payload,
                },
            )
            assert not response.is_error
            assert response.structured_content is not None
            output = response.structured_content["output"]
            assert output == expected
            decoded = IncidenceGraphResult.model_validate_json(json.dumps(output))
            cycle = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.invariant.cycle_length_profile.compute",
                    "payload": {"graph": decoded.graph.model_dump(mode="json")},
                },
            )
            assert not cycle.is_error

    asyncio.run(scenario())
