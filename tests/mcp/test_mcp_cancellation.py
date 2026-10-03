"""MCP cancellation must terminate request-owned subprocesses."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(__file__).with_name("_cancellation_server.py")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _read_pids(marker: Path) -> list[int]:
    # Generous: the marker appears only after the stdio server has started, the
    # catalog has resolved the operation, and the bounded worker has forked.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if marker.exists() and (text := marker.read_text().strip()):
            return list(json.loads(text))
        await asyncio.sleep(0.01)
    raise AssertionError("process-backed operation did not publish its PID marker")


async def _assert_pids_exit(pids: list[int]) -> None:
    # Must stay well inside the worker's own 20s run_bounded_process timeout.
    # If this deadline were longer, a surviving worker would exit on its own
    # timeout and the assertion would pass without cancellation reaping it.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if all(not _pid_exists(pid) for pid in pids):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"cancelled process tree survived: {pids}")


@pytest.mark.skipif(os.name != "posix", reason="process-tree assertion is POSIX")
def test_stdio_cancellation_reaps_tree_and_server_remains_responsive(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client, StdioServerParameters, stdio_client

        parameters = StdioServerParameters(
            command=sys.executable, args=[str(SERVER)], env=dict(os.environ), cwd=ROOT
        )
        async with Client(stdio_client(parameters), raise_exceptions=True) as client:
            for attempt in range(3):
                marker = tmp_path / f"request-{attempt}.json"
                call = asyncio.ensure_future(
                    client.call_tool(
                        "math.run",
                        {
                            "operation_id": "test.process.wait",
                            "payload": {"marker": str(marker)},
                        },
                        read_timeout_seconds=30,
                    )
                )
                pids = await _read_pids(marker)
                # Cancellation must be what ends this call. A call that had
                # already completed, or that the server aborted and cleaned up
                # on its own, would still leave the tree reaped and the
                # follow-up responsive, so neither is accepted as evidence that
                # client cancellation did the work.
                assert call.cancel(), "in-flight tool call had already finished"
                with pytest.raises(asyncio.CancelledError):
                    await call
                await _assert_pids_exit(pids)
            follow_up = await client.call_tool(
                "math.run",
                {
                    "operation_id": "integer.compute.extended_gcd",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert follow_up.structured_content["output"]["gcd"] == "6"

    asyncio.run(scenario())
