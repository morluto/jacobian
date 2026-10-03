"""MCP progress projection tests."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest
from mcp.client.session import IncomingMessage
from mcp.server.mcpserver import Context
from mcp.types import ProgressNotification

from jacobian._execution import ProgressSink
from jacobian.catalog.catalog import Catalog
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server, create_server
from jacobian.mcp.tools import _CoalescingProgressSink, run_with_mcp_progress
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


@pytest.mark.parametrize("vertices", [[], ["v"]])
@pytest.mark.parametrize("tool_name", ["math.run", "graph.chromatic_bipartition.find"])
def test_zero_work_reports_first_zero_once_and_preserves_result(
    vertices: list[str], tool_name: str
) -> None:
    async def scenario() -> None:
        operation_id = "graph.chromatic_bipartition.find"
        operation = Catalog.open().operation(operation_id)
        assert operation is not None
        catalog = Catalog((operation,))
        server = _build_server(
            state=AppState(operation_catalog=catalog),
            evaluation_tools=direct_operation_tools(catalog),
        )
        updates: list[tuple[float, float | None, str | None]] = []
        notifications: list[ProgressNotification] = []

        async def on_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            updates.append((progress, total, message))

        async def on_message(message: IncomingMessage) -> None:
            if isinstance(message, ProgressNotification):
                notifications.append(message)

        payload: dict[str, Any] = {
            "graph": {"vertices": vertices, "edges": []},
            "s": 1,
            "t": 1,
        }
        arguments = (
            {"operation_id": operation_id, "payload": payload}
            if tool_name == "math.run"
            else payload
        )
        expected = {
            **payload,
            "status": "NO_SPLIT",
            "side_a": None,
            "side_b": None,
            "chromatic_a": None,
            "chromatic_b": None,
            "checked_partitions": 0,
        }
        async with Client(server, mode="legacy", message_handler=on_message) as client:
            for requested in (True, False):
                notifications.clear()
                result = await client.call_tool(
                    tool_name,
                    arguments,
                    progress_callback=on_progress if requested else None,
                )
                assert not result.is_error
                assert isinstance(result.structured_content, dict)
                output = (
                    result.structured_content["output"]
                    if tool_name == "math.run"
                    else result.structured_content
                )
                assert output == expected
                assert len(notifications) == int(requested)
        assert updates == [(0.0, 0.0, "chromatic bipartitions checked")]

    asyncio.run(scenario())


def test_progress_bridge_drops_nonincreasing_updates_after_delivery() -> None:
    async def scenario() -> None:
        updates: list[tuple[float, float | None, str | None]] = []
        delivered: dict[float, anyio.Event] = {0: anyio.Event(), 2: anyio.Event()}

        async def report_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            updates.append((progress, total, message))
            if progress in delivered:
                delivered[progress].set()

        def report(sink: ProgressSink) -> str:
            for progress in (0, 2):
                sink.report(progress, total=4, message="accepted")
                anyio.from_thread.run(delivered[progress].wait)
                sink.report(progress, total=8, message="duplicate with new metadata")
                sink.report(progress - 1, total=8, message="decreasing")
            sink.report(4, total=4, message="complete")
            return "worker completed"

        ctx = cast(
            Context[AppState, Any], SimpleNamespace(report_progress=report_progress)
        )
        result = await run_with_mcp_progress(report, ctx)
        assert result == "worker completed"
        assert updates == [
            (0, 4, "accepted"),
            (2, 4, "accepted"),
            (4, 4, "complete"),
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("progress", [0, 2])
def test_progress_bridge_preserves_pending_update_when_counter_does_not_increase(
    progress: int,
) -> None:
    async def scenario() -> None:
        sink = _CoalescingProgressSink()
        updates: list[tuple[float, float | None, str | None]] = []

        async def report_progress(
            progress: float, total: float | None, message: str | None
        ) -> None:
            updates.append((progress, total, message))

        def report() -> None:
            sink.report(0, total=4, message="accepted")
            if progress:
                sink.report(progress, total=4, message="accepted")
            sink.report(progress, total=8, message="duplicate with new metadata")
            sink.report(progress - 1, total=8, message="decreasing")

        await anyio.to_thread.run_sync(report)
        await sink.close()
        await anyio.to_thread.run_sync(lambda: sink.report(4, message="after close"))
        ctx = cast(
            Context[AppState, Any], SimpleNamespace(report_progress=report_progress)
        )
        await sink.pump(ctx)
        assert updates == [(progress, 4, "accepted")]

    asyncio.run(scenario())
