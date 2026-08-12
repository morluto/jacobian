from __future__ import annotations

import asyncio
from pathlib import Path

from jacobian.adapters.mcp.server import create_server

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


def test_math_tool_surface_is_consistent_across_discovery(tmp_path: Path) -> None:
    server = create_server(tmp_path)
    assert server.instructions is not None
    assert "math.find" in server.instructions

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == {"math.find", "math.run"}
            assert tools["math.find"].title == "Search installed Jacobian math tools"
            assert tools["math.run"].title == "Run one installed Jacobian math tool"
            assert "authoritative runtime inventory" in (
                tools["math.find"].description or ""
            )
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

    asyncio.run(scenario())
