from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jacobian.adapters.mcp.server import create_server


@pytest.mark.requires_provider("flint")
def test_mcp_keeps_two_tools_while_a_checker_consumes_a_typed_candidate(
    mcp_state: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(mcp_state), raise_exceptions=True) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {"math.find", "math.run"}

            producer_contract = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": "matrix.normal_form.smith.compute",
                    }
                },
            )
            checker_contract = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": "matrix.normal_form.smith.verify",
                    }
                },
            )
            assert producer_contract.structured_content is not None
            assert checker_contract.structured_content is not None
            assert producer_contract.structured_content["operation"][
                "output_ports"
            ] == [{"name": "smith_form", "value_type": "SmithNormalForm"}]
            assert checker_contract.structured_content["operation"]["input_ports"] == [
                {"name": "candidate", "value_type": "SmithNormalForm"}
            ]

            computed = await client.call_tool(
                "math.run",
                {
                    "operation_id": "matrix.normal_form.smith.compute",
                    "payload": {"matrix": {"entries": [["2", "4"], ["6", "8"]]}},
                },
            )
            assert computed.structured_content is not None
            value_ref = computed.structured_content["output"]["value_refs"][
                "smith_form"
            ]

            verified = await client.call_tool(
                "math.run",
                {
                    "operation_id": "matrix.normal_form.smith.verify",
                    "payload": {
                        "input": {"matrix": {"entries": [["2", "4"], ["6", "8"]]}}
                    },
                    "inputs": {"candidate": {"value_ref": value_ref}},
                },
            )
            assert verified.structured_content is not None
            assert verified.structured_content["output"]["status"] == "VERIFIED"
            assert verified.structured_content["verification_record_uri"] is not None

    asyncio.run(scenario())
