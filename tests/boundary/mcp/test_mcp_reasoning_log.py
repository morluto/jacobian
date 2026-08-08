from __future__ import annotations

import asyncio
import json
from pathlib import Path

from jacobian.adapters.mcp.server import create_server


def test_required_reasoning_log_binds_actual_capability_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(tmp_path, reasoning_log_mode="REQUIRED"),
            raise_exceptions=True,
        ) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert set(tools) == {
                "math.find",
                "math.run",
                "reasoning.write",
            }
            assert set(tools["math.run"].input_schema["required"]) >= {
                "reasoning_run_id",
                "reasoning_call_id",
            }
            reasoning_description = tools["reasoning.write"].description
            assert reasoning_description is not None
            assert "PLAN: `phase`, `summary`" in reasoning_description
            assert "BEFORE_TOOL: `phase`, `summary`, `run_id`" in reasoning_description
            assert "AFTER_TOOL: `phase`, `summary`, `run_id`, `call_id`" in (
                reasoning_description
            )
            assert "Omit `capability_id` and `mode`" in reasoning_description
            assert "FINAL: `phase`, `summary`, `run_id`" in reasoning_description

            plan = await client.call_tool(
                "reasoning.write",
                {"phase": "PLAN", "summary": "Compute one exact integer value."},
            )
            assert isinstance(plan.structured_content, dict)
            run_id = plan.structured_content["run_id"]
            before = await client.call_tool(
                "reasoning.write",
                {
                    "phase": "BEFORE_TOOL",
                    "summary": "Use exact integer arithmetic for the gcd.",
                    "run_id": run_id,
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                },
            )
            assert isinstance(before.structured_content, dict)
            call_id = before.structured_content["call_id"]
            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "payload": {"left": "84", "right": "30"},
                    "reasoning_run_id": run_id,
                    "reasoning_call_id": call_id,
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content["execution"]["status"] == "COMPLETED"
            await client.call_tool(
                "reasoning.write",
                {
                    "phase": "AFTER_TOOL",
                    "summary": "The exact computed gcd is 6; it is not independently verified.",
                    "run_id": run_id,
                    "call_id": call_id,
                    "interpretation_status": "INTERPRETED",
                    "reported_execution_status": "COMPLETED",
                    "reported_assurance_level": "COMPUTED",
                    "reported_completeness_status": "COMPLETE",
                },
            )
            await client.call_tool(
                "reasoning.write",
                {
                    "phase": "FINAL",
                    "summary": "The bounded integer computation completed with computed assurance.",
                    "run_id": run_id,
                },
            )
            resource = await client.read_resource(f"reasoning://run/{run_id}")
            events = [
                json.loads(line) for line in resource.contents[0].text.splitlines()
            ]
            finished = next(
                event for event in events if event["kind"] == "CAPABILITY_FINISHED"
            )
            assert finished["payload"]["execution_status"] == "COMPLETED"
            assert (
                finished["payload"]["assurance"]
                == result.structured_content["assurance"]
            )
            after = next(event for event in events if event["kind"] == "AFTER_TOOL")
            assert after["payload"]["execution_status_matches"] is True
            assert after["payload"]["assurance_level_matches"] is True
            assert after["payload"]["completeness_status_matches"] is True
            encoded = json.dumps(events)
            assert '"left": "84"' not in encoded
            assert '"gcd": "6"' not in encoded

    asyncio.run(scenario())


def test_off_mode_preserves_the_legacy_two_tool_surface(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(tmp_path, reasoning_log_mode="OFF"),
            raise_exceptions=True,
        ) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert set(tools) == {"math.find", "math.run"}
            assert set(tools["math.run"].input_schema["properties"]) == {
                "capability_id",
                "payload",
                "mode",
            }

    asyncio.run(scenario())


def test_audit_mode_allows_unbound_legacy_invocation(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(tmp_path, reasoning_log_mode="AUDIT"),
            raise_exceptions=True,
        ) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content["execution"]["status"] == "COMPLETED"

    asyncio.run(scenario())


def test_required_mode_rejects_mismatched_call_binding_without_consuming_it(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(tmp_path, reasoning_log_mode="REQUIRED"),
            raise_exceptions=True,
        ) as client:
            plan = await client.call_tool(
                "reasoning.write",
                {"phase": "PLAN", "summary": "Compute one exact value."},
            )
            assert isinstance(plan.structured_content, dict)
            run_id = plan.structured_content["run_id"]
            before = await client.call_tool(
                "reasoning.write",
                {
                    "phase": "BEFORE_TOOL",
                    "summary": "Bind the intended integer capability.",
                    "run_id": run_id,
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                },
            )
            assert isinstance(before.structured_content, dict)
            call_id = before.structured_content["call_id"]

            mismatch = await client.call_tool(
                "math.run",
                {
                    "capability_id": "integer.factorization.factor",
                    "mode": "EXPLORE",
                    "payload": {"value": "84"},
                    "reasoning_run_id": run_id,
                    "reasoning_call_id": call_id,
                },
            )
            assert mismatch.is_error is True
            assert "REASONING_CALL_MISMATCH" in mismatch.content[0].text

            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "payload": {"left": "84", "right": "30"},
                    "reasoning_run_id": run_id,
                    "reasoning_call_id": call_id,
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content["execution"]["status"] == "COMPLETED"

    asyncio.run(scenario())
