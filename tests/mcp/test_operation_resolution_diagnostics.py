"""Unknown IDs preserve each MCP failure signal and a shared recovery diagnostic."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.types import TextContent

from jacobian.mcp.server import create_server
from mcp import Client


@pytest.mark.parametrize("operation_id", ["missing.operation", "missing." + "x" * 120])
def test_unknown_operation_diagnostics_and_same_client_recovery(
    operation_id: str,
) -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            inspected = await client.call_tool(
                "math.find", {"operation_id": operation_id}
            )
            assert inspected.is_error is False
            assert isinstance(inspected.structured_content, dict)
            assert inspected.structured_content["kind"] == "error"
            diagnostic = inspected.structured_content["error"]
            assert diagnostic["code"] == "UNKNOWN_OPERATION"
            assert diagnostic["stage"] == "operation_resolution"
            assert operation_id in diagnostic["message"]
            assert "math.find" in diagnostic["hint"]
            assert isinstance(inspected.content[0], TextContent)
            assert json.loads(inspected.content[0].text) == inspected.structured_content
            assert len(inspected.content[0].text.encode("utf-8")) < 2_048

            executed = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": {}}
            )
            assert executed.is_error is True
            assert executed.structured_content is None
            assert isinstance(executed.content[0], TextContent)
            assert len(executed.content[0].text.encode("utf-8")) < 2_048
            assert (
                json.loads(
                    executed.content[0].text.removeprefix(
                        "Error executing tool math.run: "
                    )
                )
                == diagnostic
            )

            unmatched = await client.call_tool("math.find", {"query": "zqx"})
            assert unmatched.is_error is False
            assert isinstance(unmatched.structured_content, dict)
            assert unmatched.structured_content["kind"] == "matches"
            assert unmatched.structured_content["matches"] == []
            assert unmatched.structured_content["total_matches"] == 0
            assert unmatched.structured_content["next_cursor"] is None

            recovered_inspection = await client.call_tool(
                "math.find", {"operation_id": "integer.compute.extended_gcd"}
            )
            assert recovered_inspection.is_error is False
            assert isinstance(recovered_inspection.structured_content, dict)
            assert recovered_inspection.structured_content["kind"] == "operation"
            assert (
                recovered_inspection.structured_content["operation"]["operation_id"]
                == "integer.compute.extended_gcd"
            )
            recovered_execution = await client.call_tool(
                "math.run",
                {
                    "operation_id": "integer.compute.extended_gcd",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert recovered_execution.is_error is False
            assert isinstance(recovered_execution.structured_content, dict)
            assert recovered_execution.structured_content["output"] == {
                "gcd": "6",
                "left_coefficient": "-1",
                "right_coefficient": "3",
            }

    asyncio.run(scenario())
