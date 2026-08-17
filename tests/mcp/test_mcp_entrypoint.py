"""MCP process entrypoint surface: help, version, and stdio tool names."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

MCP_TOOL_NAMES = {
    "math.find",
    "math.run",
}


def test_mcp_stdio_entrypoint_exposes_stable_math_tools() -> None:
    async def scenario() -> None:
        from mcp import Client, StdioServerParameters, stdio_client

        environment = dict(os.environ)
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "jacobian.mcp"],
            env=environment,
            cwd=Path.cwd(),
        )
        async with Client(
            stdio_client(parameters),
            raise_exceptions=True,
        ) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == MCP_TOOL_NAMES

    asyncio.run(scenario())


def test_mcp_entrypoint_has_nonstarting_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "jacobian.mcp", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0
    assert "Run one local Jacobian MCP server over stdio" in completed.stdout
    assert "--transport" not in completed.stdout
    assert "--auth-tokens-file" not in completed.stdout
    assert "--tool-profile" not in completed.stdout
    assert "--tool-name-profile" not in completed.stdout
    assert "--reasoning-log-mode" not in completed.stdout


def test_mcp_entrypoint_reports_distribution_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "jacobian.mcp", "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == f"jacobian-mcp {version('jacobian')}"
