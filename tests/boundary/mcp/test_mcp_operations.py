from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from importlib.metadata import version
from pathlib import Path

import pytest

from jacobian.adapters.mcp.context import _public_tool_error
from jacobian.adapters.mcp.server import create_server
from jacobian.adapters.mcp.tooling import _request_trace_digest, _run_blocking

MCP_TOOL_NAMES = {
    "math.find",
    "math.run",
    "reasoning.write",
}


def test_mcp_trace_correlation_hashes_headers_without_retaining_them() -> None:
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

    class RequestContext:
        def __init__(self) -> None:
            self.headers = {"traceparent": traceparent}
            self.request_id = "private-request-id"

    digest, source = _request_trace_digest(RequestContext())

    assert digest == hashlib.sha256(traceparent.encode()).hexdigest()[:8]
    assert source == "traceparent"
    assert traceparent not in digest
    assert "private-request-id" not in digest


def test_mcp_logs_bounded_tool_metrics_without_arguments(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jacobian.adapters.mcp.server")
    caplog.set_level(logging.INFO, logger="jacobian.adapters.mcp.tooling")

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            await client.call_tool(
                "math.find",
                {"query": "private-query-marker"},
            )
            failed = await client.call_tool(
                "math.run",
                {
                    "capability_id": "missing.capability",
                    "mode": "EXPLORE",
                    "payload": {"private": "private-payload-marker"},
                },
            )
            response = json.loads(failed.content[0].text)
            assert response["execution"]["status"] == "ERROR"

    asyncio.run(scenario())

    messages = [record.getMessage() for record in caplog.records]
    metric = next(
        message
        for message in messages
        if "MCP tool call tool=math.find status=success" in message
    )
    assert "duration_ms=" in metric
    assert "response_bytes=" in metric
    assert "argument_digest=sha256:" in metric
    assert "private-query-marker" not in metric
    attempt = next(
        message
        for message in messages
        if "MCP capability attempt" in message
        and "capability_id=missing.capability" in message
    )
    assert "execution_status=ERROR" in attempt
    assert "diagnostic_codes=UNKNOWN_CAPABILITY" in attempt
    assert "trace_digest=" in attempt
    assert "argument_digest=sha256:" in attempt
    assert "private-payload-marker" not in attempt


def test_mcp_tool_failures_return_safe_actionable_errors(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=False) as client:
            unknown_capability = await client.call_tool(
                "math.find", {"capability_id": "missing.capability"}
            )
            response = json.loads(unknown_capability.content[0].text)
            assert response["error"]["code"] == "UNKNOWN_CAPABILITY"
            assert "search installed capabilities" in response["error"]["hint"]

    asyncio.run(scenario())

    internal = json.loads(_public_tool_error("fixture", KeyError("internal")))
    assert internal["error"]["code"] == "OPERATION_FAILED"


def test_mcp_protocol_and_authentication_errors_remain_distinct(tmp_path: Path) -> None:
    from mcp.shared.exceptions import MCPError

    server = create_server(tmp_path)

    @server.tool(name="fixture.protocol-error")
    async def protocol_error() -> None:
        raise MCPError(123, "protocol action required")

    with pytest.raises(MCPError, match="protocol action required"):
        asyncio.run(server.call_tool("fixture.protocol-error", {}))

    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(
                tmp_path,
                tenant_isolation=True,
                allow_anonymous=False,
            ),
            raise_exceptions=False,
        ) as client:
            with pytest.raises(MCPError) as authentication_error:
                await client.call_tool("math.find", {})
            assert str(authentication_error.value) == "Internal server error"

    asyncio.run(scenario())


def test_direct_tool_calls_reject_removed_and_malformed_arguments(
    tmp_path: Path,
) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    async def scenario() -> None:
        server = create_server(tmp_path)

        with pytest.raises(ToolError):
            await server.call_tool("workspace.write", {})

        with pytest.raises(ToolError):
            await server.call_tool(
                "math.find",
                {"limit": "not-an-integer"},
            )

    asyncio.run(scenario())


def test_mcp_stdio_entrypoint_exposes_required_reasoning_tool(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client, StdioServerParameters, stdio_client

        environment = dict(os.environ)
        environment["JACOBIAN_STATE_DIR"] = str(tmp_path)
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "jacobian.adapters.mcp.server",
                "--reasoning-log-mode",
                "required",
            ],
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
        [sys.executable, "-m", "jacobian.adapters.mcp.server", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0
    assert "Run the Jacobian MCP server" in completed.stdout
    assert "--tool-profile" not in completed.stdout
    assert "--tool-name-profile" not in completed.stdout


def test_mcp_entrypoint_reports_distribution_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "jacobian.adapters.mcp.server", "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == f"jacobian-mcp {version('jacobian')}"


def test_cancelled_mcp_blocking_work_drains_before_request_task_finishes() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    cancellation_signalled = threading.Event()

    def blocking_operation() -> str:
        started.set()
        release.wait(timeout=2)
        finished.set()
        return "finished"

    async def scenario() -> None:
        task = asyncio.create_task(
            _run_blocking(
                blocking_operation,
                on_cancel=cancellation_signalled.set,
            )
        )
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.05)
        assert cancellation_signalled.is_set()
        assert not task.done()
        assert not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(scenario())
