from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp import Client

from jacobian.adapters.mcp.server import create_server


def test_reasoning_run_survives_runtime_restart_and_finalizes(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with Client(
            create_server(tmp_path, reasoning_log_mode="REQUIRED"),
            raise_exceptions=True,
        ) as client:
            plan = await client.call_tool(
                "reasoning.write",
                {"phase": "PLAN", "summary": "Compute and audit an exact gcd."},
            )
            assert isinstance(plan.structured_content, dict)
            run_id = plan.structured_content["run_id"]
            before = await client.call_tool(
                "reasoning.write",
                {
                    "phase": "BEFORE_TOOL",
                    "summary": "Use exact integer arithmetic.",
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
                    "payload": {"left": "84", "right": "30"},
                    "mode": "EXPLORE",
                    "reasoning_run_id": run_id,
                    "reasoning_call_id": call_id,
                },
            )
            assert isinstance(result.structured_content, dict)
            await client.call_tool(
                "reasoning.write",
                {
                    "phase": "AFTER_TOOL",
                    "summary": "The exact computation returned gcd 6 with computed assurance.",
                    "run_id": run_id,
                    "call_id": call_id,
                    "interpretation_status": "INTERPRETED",
                    "reported_execution_status": "COMPLETED",
                    "reported_assurance_level": "COMPUTED",
                    "reported_completeness_status": "COMPLETE",
                },
            )

        async with Client(
            create_server(tmp_path, reasoning_log_mode="REQUIRED"),
            raise_exceptions=True,
        ) as restarted:
            final = await restarted.call_tool(
                "reasoning.write",
                {
                    "phase": "FINAL",
                    "summary": "The finite computation is complete; no wider claim is made.",
                    "run_id": run_id,
                },
            )
            assert isinstance(final.structured_content, dict)
            assert final.structured_content["state"] == "FINALIZED"
            resource = await restarted.read_resource(f"reasoning://run/{run_id}")
            events = [
                json.loads(line) for line in resource.contents[0].text.splitlines()
            ]
            assert events[-1]["kind"] == "FINAL"
            assert [event["sequence"] for event in events] == list(range(len(events)))

    asyncio.run(scenario())
