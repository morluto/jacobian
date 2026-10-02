"""Graph inspection publishes pair normalization at the actual MCP boundary."""

from __future__ import annotations

import asyncio
import json

from mcp.types import ContentBlock, TextContent

from jacobian.mcp.server import create_server
from mcp import Client


def _content_text(block: ContentBlock) -> str:
    assert isinstance(block, TextContent)
    return block.text


def test_graph_inspection_and_execution_accept_either_undirected_endpoint_order() -> (
    None
):
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
        async with Client(create_server(), raise_exceptions=False) as client:
            inspected = await client.call_tool(
                "math.find",
                {"operation_id": operation_id},
            )
            contract = inspected.structured_content["operation"]
            input_schema = contract["input_schema"]
            reference = input_schema["properties"]["graph"]["$ref"]
            schema = input_schema["$defs"][reference.rsplit("/", 1)[1]]
            description = schema["properties"]["edges"]["description"]
            assert "lexicographic label order" in description
            assert "either endpoint order" in description
            assert "list order are preserved" in description
            example = contract["examples"][0]["input"]
            assert example["graph"]["vertices"] != sorted(example["graph"]["vertices"])
            example_result = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": example}
            )
            assert example_result.structured_content["output"]["clique_count"] == 2

            result = await client.call_tool(
                "math.run",
                {"operation_id": operation_id, "payload": {"graph": graph}},
            )
            output = result.structured_content["output"]
            assert output["graph"] == {
                **graph,
                "edges": [sorted(edge) for edge in edges],
            }
            assert output["clique_count"] == 4
            assert {tuple(members) for _, members in output["hypergraph"]["edges"]} == {
                ("r", "x", "y", "z"),
                ("u", "x", "y", "z"),
                ("u", "w", "x", "y"),
                ("u", "v", "x"),
            }

            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": operation_id,
                    "payload": {"graph": {**graph, "edges": [["x", "r"], ["r", "x"]]}},
                },
            )
            error = json.loads(
                _content_text(rejected.content[0]).removeprefix(
                    "Error executing tool math.run: "
                )
            )["errors"][0]
            assert error["location"] == ["graph"]
            assert error["code"] == "graph.graph_edges_must_be_unique"

    asyncio.run(scenario())
