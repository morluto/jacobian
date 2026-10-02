"""MCP exposes the scalar input alternatives and preserves real output handoffs."""

import asyncio
import json

from mcp.types import TextContent

from jacobian.mcp.server import create_server
from mcp import Client


def test_mcp_integer_ingress_inspection_composition_and_recovery() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            inspected = await client.call_tool(
                "math.find", {"operation_id": "integer.compute.nth_prime"}
            )
            assert not inspected.is_error
            assert inspected.structured_content is not None
            contract = inspected.structured_content["operation"]
            scalar = contract["input_schema"]["properties"]["n"]
            assert {branch["type"] for branch in scalar["anyOf"]} == {
                "string",
                "integer",
            }
            assert "[1, 10000]" in scalar["description"]
            assert contract["examples"][0]["input"] == {"n": "6"}

            produced = await client.call_tool(
                "math.run",
                {
                    "operation_id": "integer.compute.gcd",
                    "payload": {"left": 12, "right": "18"},
                },
            )
            assert not produced.is_error
            assert produced.structured_content is not None
            scalar_output = produced.structured_content["output"]["value"]
            assert scalar_output == "6"
            for operation, expected in (
                ("euler_totient", {"value": "2"}),
                ("divisor_count", {"value": "4"}),
                ("divisor_sum", {"value": "12"}),
                ("mobius", {"value": "1"}),
                ("next_prime", {"value": "7"}),
                ("nth_prime", {"value": "13"}),
                ("prime_count", {"value": "3"}),
                ("previous_prime", {"value": "5"}),
                ("primorial", {"value": "30030"}),
                ("floor_square_root", {"root": 2}),
            ):
                result = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "integer.compute." + operation,
                        "payload": {"n": scalar_output},
                    },
                )
                assert not result.is_error
                assert result.structured_content is not None
                assert result.structured_content["output"] == expected

            for invalid in (True, 6.5, "06", "+6", "-0", "6\n", "10001", "1" * 1000):
                refused = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "integer.compute.euler_totient",
                        "payload": {"n": invalid},
                    },
                )
                assert refused.is_error
                assert isinstance(refused.content[0], TextContent)
                text = refused.content[0].text
                diagnostic = json.loads(text[text.index("{") :])
                assert diagnostic["code"] == "INVALID_REQUEST"
                if not isinstance(invalid, float):
                    assert len(diagnostic["errors"]) == 1
                    error = diagnostic["errors"][0]
                    assert error["location"] == ["n"]
                    assert error["code"] == "number_theory.integer_input"
            recovered = await client.call_tool(
                "math.run",
                {
                    "operation_id": "integer.compute.euler_totient",
                    "payload": {"n": scalar_output},
                },
            )
            assert not recovered.is_error
            assert recovered.structured_content is not None
            assert recovered.structured_content["output"] == {"value": "2"}

    asyncio.run(scenario())
