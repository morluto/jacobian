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
from jacobian.math.graphs.cycle_length_profile._models import CycleLengthProfileRequest
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.mcp.server import create_server
from mcp import Client


def test_incidence_graph_round_trips_into_cycle_profile() -> None:
    async def scenario() -> None:
        payload = {
            "hypergraph": {
                "vertices": ["a", "b", "isolated"],
                "edges": [["e1", ["a", "b"]], ["e2", ["a", "b"]]],
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
            assert cycle.structured_content is not None
            assert cycle.structured_content["output"]["rows"][0]["cycle_length"] == 4

            boundary_payload = {
                "hypergraph": {
                    "vertices": [str(index) for index in range(256)],
                    "edges": [],
                }
            }
            boundary = await client.call_tool(
                "math.run",
                {
                    "operation_id": "hypergraph.incidence_graph.compute",
                    "payload": boundary_payload,
                },
            )
            assert not boundary.is_error
            assert boundary.structured_content is not None
            boundary_graph = boundary.structured_content["output"]["graph"]
            assert len(boundary_graph["vertices"]) == 256
            consumed_boundary = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.invariant.cycle_length_profile.compute",
                    "payload": {"graph": boundary_graph},
                },
            )
            assert not consumed_boundary.is_error
            assert consumed_boundary.structured_content is not None
            assert consumed_boundary.structured_content["output"]["rows"] == []

            over_payload = {
                "hypergraph": {
                    "vertices": [str(index) for index in range(256)],
                    "edges": [["edge", []]],
                }
            }
            over = await client.call_tool(
                "math.run",
                {
                    "operation_id": "hypergraph.incidence_graph.compute",
                    "payload": over_payload,
                },
            )
            assert not over.is_error
            assert over.structured_content is not None
            over_graph = over.structured_content["output"]["graph"]
            assert len(over_graph["vertices"]) == 257
            assert SimpleUndirectedGraph.model_validate(over_graph)
            CycleLengthProfileRequest.model_validate({"graph": over_graph})
            consumed_over = await client.call_tool(
                "math.run",
                {
                    "operation_id": "graph.invariant.cycle_length_profile.compute",
                    "payload": {"graph": over_graph},
                },
            )
            assert consumed_over.is_error

    asyncio.run(scenario())
