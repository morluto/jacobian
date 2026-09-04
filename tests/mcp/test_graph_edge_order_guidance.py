"""Graph inspection and validation explain how to orient undirected edges."""

from __future__ import annotations

import asyncio

import pytest
from mcp.shared.exceptions import MCPError

from jacobian.mcp.server import create_server
from mcp import Client


def test_graph_inspection_and_error_explain_endpoint_order_before_recovery() -> None:
    async def scenario() -> None:
        operation_id = "graph.maximal_clique_hypergraph.construct"
        edges = [
            ["x", "y"],
            ["x", "z"],
            ["x", "r"],
            ["y", "z"],
            ["y", "r"],
            ["z", "r"],
            ["x", "u"],
            ["y", "u"],
            ["z", "u"],
            ["x", "w"],
            ["y", "w"],
            ["u", "w"],
            ["x", "v"],
            ["u", "v"],
        ]
        graph = {
            "vertices": ["x", "y", "z", "r", "u", "v", "w"],
            "edges": edges,
        }
        async with Client(create_server(), raise_exceptions=True) as client:
            inspected = await client.call_tool(
                "math.find",
                {"request": {"op": "inspect", "operation_id": operation_id}},
            )
            contract = inspected.structured_content["operation"]
            schema = contract["input_schema"]["$defs"]["SimpleUndirectedGraph"]
            description = schema["properties"]["edges"]["description"]
            assert "lexicographic label order" in description
            assert "not positions in vertices" in description
            example = contract["examples"][0]["input"]
            assert example["graph"]["vertices"] != sorted(example["graph"]["vertices"])
            example_result = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": example}
            )
            assert example_result.structured_content["output"]["clique_count"] == 2

            with pytest.raises(MCPError) as rejected:
                await client.call_tool(
                    "math.run",
                    {"operation_id": operation_id, "payload": {"graph": graph}},
                )
            error = rejected.value.data["errors"][0]
            assert error["location"] == ["graph"]
            assert "lexicographic label order" in error["message"]
            assert "not positions in vertices" in error["message"]

            graph["edges"] = [sorted(edge) for edge in edges]
            result = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": {"graph": graph}}
            )
            output = result.structured_content["output"]
            assert output["graph"] == graph
            assert output["clique_count"] == 4
            assert {tuple(members) for _, members in output["hypergraph"]["edges"]} == {
                ("r", "x", "y", "z"),
                ("u", "x", "y", "z"),
                ("u", "w", "x", "y"),
                ("u", "v", "x"),
            }

    asyncio.run(scenario())
