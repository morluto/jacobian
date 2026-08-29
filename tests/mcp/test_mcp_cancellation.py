"""MCP cancellation must terminate request-owned subprocesses."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError

ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(__file__).with_name("_cancellation_server.py")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _read_pids(marker: Path) -> list[int]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if marker.exists() and (text := marker.read_text().strip()):
            return list(json.loads(text))
        await asyncio.sleep(0.01)
    raise AssertionError("process-backed operation did not publish its PID marker")


async def _assert_pids_exit(pids: list[int]) -> None:
    deadline = time.monotonic() + 4
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
                with pytest.raises(MCPError, match="timed out"):
                    await client.call_tool(
                        "test.process.wait",
                        {"marker": str(marker)},
                        read_timeout_seconds=2,
                    )
                await _assert_pids_exit(await _read_pids(marker))
            follow_up = await client.call_tool(
                "integer.compute.extended_gcd",
                {"left": "84", "right": "30"},
            )
            assert follow_up.structured_content["gcd"] == "6"

    asyncio.run(scenario())
