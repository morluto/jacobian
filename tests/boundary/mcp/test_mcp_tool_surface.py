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
            assert tools["math.find"].title == "Find an exact mathematical operation"
            assert tools["math.run"].title == "Run a mathematical operation"
            assert "math.find" in (tools["math.run"].description or "")

            described = await client.call_tool(
                "math.find",
                {"capability_id": "integer.compute.gcd", "view": "CONTRACT"},
            )
            assert described.structured_content is not None
            invocations = described.structured_content["invocations"]
            assert invocations
            assert {item["tool"] for item in invocations} == {"math.run"}

            absent = await client.call_tool(
                "math.find",
                {"query": "quuxonium frobnicator"},
            )
            assert absent.structured_content is not None
            recovery_tools = {
                item["tool"]
                for item in absent.structured_content["available_recovery_paths"]
                if "tool" in item
            }
            assert recovery_tools == {"math.find"}

    asyncio.run(scenario())
