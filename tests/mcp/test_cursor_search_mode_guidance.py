"""Discovery cursors document and enforce their search-mode identity."""

import asyncio

from jacobian.mcp.server import create_server
from mcp import Client


def test_cursor_guidance_and_search_mode_recovery() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            tools = await client.list_tools()
            find = next(tool for tool in tools.tools if tool.name == "math.find")
            assert find.description is not None
            assert "query, namespace, and search_mode" in find.description
            description = find.input_schema["properties"]["cursor"]["description"]
            assert "query, namespace, and search_mode" in description
            query = "exact determinant of a rational matrix"
            first = await client.call_tool(
                "math.find", {"query": query, "limit": 1, "search_mode": "precise"}
            )
            assert first.structured_content is not None
            cursor = first.structured_content["next_cursor"]
            assert cursor is not None
            changed = await client.call_tool(
                "math.find",
                {"query": query, "limit": 1, "search_mode": "broad", "cursor": cursor},
            )
            assert changed.structured_content is not None
            error = changed.structured_content["error"]
            assert error["code"] == "INVALID_CURSOR"
            assert "query, namespace, and search_mode" in error["hint"]
            assert "without a cursor" in error["hint"]
            restored = await client.call_tool(
                "math.find",
                {
                    "query": query,
                    "limit": 2,
                    "search_mode": "precise",
                    "cursor": cursor,
                },
            )
            assert restored.structured_content is not None
            assert restored.structured_content["kind"] == "matches"
            assert restored.structured_content["search_mode"] == "precise"
            assert restored.structured_content["matches"]
            assert first.structured_content["matches"][0]["operation_id"] not in {
                item["operation_id"] for item in restored.structured_content["matches"]
            }
            restarted = await client.call_tool(
                "math.find", {"query": query, "limit": 1, "search_mode": "broad"}
            )
            assert restarted.structured_content is not None
            assert restarted.structured_content["kind"] == "matches"
            assert restarted.structured_content["search_mode"] == "broad"
            assert restarted.structured_content["matches"]

    asyncio.run(scenario())
