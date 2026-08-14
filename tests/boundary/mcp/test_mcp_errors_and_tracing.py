"""MCP error surfaces, recovery hints, and bounded tracing metrics."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from jacobian.adapters.mcp.context import _public_tool_error
from jacobian.adapters.mcp.remote import create_remote_server
from jacobian.adapters.mcp.server import create_server
from jacobian.domains.number_theory import number_theory_operations
from tests.boundary.mcp.mcp_support import open_focused_mcp_server


def test_mcp_logs_bounded_operation_metrics_without_arguments(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jacobian.adapters.mcp.tooling")

    async def scenario() -> None:
        from mcp import Client

        with open_focused_mcp_server(
            tmp_path,
            number_theory_operations(),
        ) as server:
            async with Client(server, raise_exceptions=True) as client:
                await client.call_tool(
                    "math.find",
                    {
                        "request": {
                            "op": "search",
                            "query": "private-query-marker",
                        }
                    },
                )
                failed = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "missing.operation",
                        "payload": {"private": "private-payload-marker"},
                    },
                )
                response = json.loads(failed.content[0].text)
                assert response["execution"]["status"] == "ERROR"

    asyncio.run(scenario())

    messages = [record.getMessage() for record in caplog.records]
    attempt = next(
        message
        for message in messages
        if "MCP operation attempt" in message
        and "operation_id=missing.operation" in message
    )
    assert "execution_status=ERROR" in attempt
    assert "diagnostic_codes=UNKNOWN_OPERATION" in attempt
    assert "argument_digest=sha256:" in attempt
    assert "provider=unknown" in attempt
    assert "checker_ids=none" in attempt
    assert "artifact_count=0" in attempt
    assert "trace_digest=" not in attempt
    assert "duration_ms=" not in attempt
    assert "private-payload-marker" not in attempt
    assert "private-query-marker" not in " ".join(messages)


def test_mcp_tool_failures_return_safe_actionable_errors(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        with open_focused_mcp_server(tmp_path) as server:
            async with Client(server, raise_exceptions=False) as client:
                unknown_operation = await client.call_tool(
                    "math.find",
                    {
                        "request": {
                            "op": "inspect",
                            "operation_id": "missing.operation",
                        }
                    },
                )
                response = json.loads(unknown_operation.content[0].text)
                assert response["error"]["code"] == "UNKNOWN_OPERATION"
                assert "search installed operations" in response["error"]["hint"]
                assert "available_operation_ids" not in response["error"]
                assert isinstance(unknown_operation.structured_content, dict)
                error = unknown_operation.structured_content["error"]
                assert len(error["nearby_operation_ids"]) <= 5
                assert error["available_recovery_paths"][-1] == {
                    "action": "inspect_catalog",
                    "resource_uri": "operation://catalog",
                }
                assert len(json.dumps(error).encode("utf-8")) < 2_048

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
            create_remote_server(
                tmp_path,
                allow_anonymous=False,
            ),
            raise_exceptions=False,
        ) as client:
            authentication_error = await client.call_tool("math.find", {})
            assert authentication_error.is_error is True

    asyncio.run(scenario())


def test_direct_tool_calls_reject_removed_and_malformed_arguments(
    tmp_path: Path,
) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    async def scenario() -> None:
        with open_focused_mcp_server(tmp_path) as server:
            with pytest.raises(ToolError):
                await server.call_tool("workspace.write", {})

            with pytest.raises(ToolError):
                await server.call_tool(
                    "math.find",
                    {
                        "request": {
                            "op": "search",
                            "query": "matrix",
                            "limit": "not-an-integer",
                        }
                    },
                )

    asyncio.run(scenario())
