"""Actual SDK projections agree on the owner-declared result wire aliases."""

from __future__ import annotations

import asyncio
import json

from jsonschema import Draft202012Validator
from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.mcp.direct_tools import _direct_operation_tool
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client


def test_mcp_pauli_output_matches_schema_direct_tool_and_unchanged_consumer() -> None:
    async def scenario() -> None:
        catalog = Catalog.open()
        operation_id = "quantum.pauli.qubit.from_labels.compute"
        operation = catalog.operation(operation_id)
        assert operation is not None
        server = _build_server(
            state=AppState(operation_catalog=catalog),
            evaluation_tools=[_direct_operation_tool(operation, catalog)],
        )
        payload = {
            "register": {"qubit_ids": ["left", "right"]},
            "labels": ["I", "Y"],
            "phase": 2,
        }
        async with Client(server, raise_exceptions=True) as client:
            inspected = await client.call_tool(
                "math.find", {"operation_id": operation_id}
            )
            produced = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": payload}
            )
            assert produced.is_error is False
            assert isinstance(inspected.structured_content, dict)
            assert isinstance(produced.structured_content, dict)
            output = produced.structured_content["output"]
            Draft202012Validator(
                inspected.structured_content["operation"]["output_schema"]
            ).validate(output)
            assert output["source"] == payload
            assert output["pauli"]["phase_free"]["register"] == payload["register"]
            text = produced.content[0]
            assert isinstance(text, TextContent)
            assert json.loads(text.text)["output"] == output

            direct = await client.call_tool(operation_id, payload)
            assert direct.is_error is False
            assert direct.structured_content == output
            recovered = await client.call_tool(
                "math.run",
                {
                    "operation_id": "quantum.pauli.qubit.to_labels.compute",
                    "payload": {"pauli": output["pauli"]},
                },
            )
            assert isinstance(recovered.structured_content, dict)
            assert recovered.structured_content["output"]["labels"] == payload["labels"]
            assert recovered.structured_content["output"]["phase"] == payload["phase"]

    asyncio.run(scenario())
