"""MCP progress projection tests."""

import asyncio

from jacobian.mcp.server import create_server
from mcp import Client


def test_exact_cover_reports_count_only_progress_when_requested() -> None:
    async def scenario() -> None:
        updates: list[tuple[float, float | None, str | None]] = []

        async def on_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            updates.append((progress, total, message))

        async with Client(create_server()) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "combinatorics.generalized_exact_cover.find",
                    "payload": {
                        "instance": {
                            "primary_items": ["p"],
                            "secondary_items": [],
                            "rows": [
                                {"row_id": "a", "items": ["p"]},
                                {"row_id": "b", "items": ["p"]},
                            ],
                        },
                        "search_node_limit": 1,
                    },
                },
                progress_callback=on_progress,
            )
        assert not result.is_error
        assert updates == [(1.0, None, "exact-cover search nodes visited")]

    asyncio.run(scenario())
