"""Closed discrepancy decisions and exact evaluator composition through MCP."""

import asyncio
import json

from jsonschema import validate

from jacobian.math.combinatorics.discrepancy.bounded_coloring._tools import TOOLS
from jacobian.mcp.server import create_server
from mcp import Client


def test_bounded_coloring_native_mcp_statuses_schema_and_eval() -> None:
    async def scenario() -> None:
        tool = TOOLS[0]
        payloads = [example.input for example in tool.examples]
        async with Client(create_server(), raise_exceptions=False) as client:
            rejected = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": {
                        "set_system": {"ground_set_size": 0, "sets": [[]]},
                        "absolute_bounds": [1],
                    },
                },
            )
            assert rejected.is_error
            statuses = []
            for payload in payloads:
                validate(payload, tool.request_type.model_json_schema())
                request = tool.request_type.model_validate_json(json.dumps(payload))
                native = tool.run(request).model_dump(mode="json")
                response = await client.call_tool(
                    "math.run", {"operation_id": tool.operation_id, "payload": payload}
                )
                assert not response.is_error
                assert response.structured_content is not None
                output = response.structured_content["output"]
                assert output == native
                validate(output, tool.result_type.model_json_schema())
                outcome = output["outcome"]
                statuses.append(outcome["status"])
                if outcome["status"] == "SATISFIABLE":
                    evaluated = await client.call_tool(
                        "math.run",
                        {
                            "operation_id": "discrepancy.theory.eval.compute",
                            "payload": {
                                "set_system": output["set_system"],
                                "coloring": outcome["coloring"],
                            },
                        },
                    )
                    assert not evaluated.is_error
                    assert evaluated.structured_content is not None
                    assert (
                        evaluated.structured_content["output"]["signed_sums"]
                        == outcome["signed_sums"]
                    )
                else:
                    assert set(outcome) == {"status"}
            assert statuses == ["SATISFIABLE", "UNSATISFIABLE"]
            exhausted = await client.call_tool(
                "math.run",
                {
                    "operation_id": tool.operation_id,
                    "payload": {
                        "set_system": {
                            "ground_set_size": 6,
                            "sets": [list(range(6))],
                        },
                        "absolute_bounds": [0],
                        "resource_budget": {"solver_work_limit": 1},
                    },
                },
            )
            assert exhausted.is_error
            assert exhausted.structured_content is None

    asyncio.run(scenario())
