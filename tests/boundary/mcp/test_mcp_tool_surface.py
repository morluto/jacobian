from __future__ import annotations

import asyncio
from pathlib import Path

from jacobian.domains.number_theory import build_number_theory_bundle
from tests.boundary.mcp.mcp_support import open_focused_mcp_server

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


def test_math_tool_surface_is_consistent_across_discovery(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == {"math.find", "math.run"}
            assert tools["math.find"].title == "Find an exact mathematical operation"
            assert tools["math.run"].title == "Run a mathematical operation"
            assert "math.find" in (tools["math.run"].description or "")

            described = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "capability_id": "integer.compute.gcd",
                    }
                },
            )
            assert described.structured_content is not None
            examples = described.structured_content["capability"]["invocation_examples"]
            assert examples

            absent = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "quuxonium frobnicator",
                    }
                },
            )
            assert absent.structured_content is not None
            assert absent.structured_content["catalog_resource"] == (
                "capability://catalog"
            )

    with open_focused_mcp_server(
        tmp_path,
        build_number_theory_bundle(),
    ) as server:
        assert server.instructions is not None
        assert "math.find" in server.instructions
        asyncio.run(scenario())
