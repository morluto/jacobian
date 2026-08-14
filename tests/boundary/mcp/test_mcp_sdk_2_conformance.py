"""Executable conformance checks for the pinned MCP Python SDK boundary."""

from __future__ import annotations

import asyncio
import importlib.metadata
from pathlib import Path

import pytest
from mcp_types.methods import serialize_server_result

import jacobian.adapters.mcp.server as server_module
from jacobian.adapters.mcp.server import create_server
from jacobian.contracts.operations import OperationResult


def test_mcp_sdk_is_exactly_pinned_and_v2_bindings_are_used() -> None:
    assert importlib.metadata.version("mcp") == "2.0.0"
    assert importlib.metadata.version("mcp-types") == "2.0.0"


def test_mcp_v2_static_validation_context_errors_and_structured_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(server_module, "Context", raising=False)

    async def scenario() -> None:
        from mcp import Client
        from mcp.shared.exceptions import MCPError

        server = create_server(
            tmp_path,
        )
        assert not hasattr(server_module, "Context")
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            assert all(
                tool.input_schema.get("additionalProperties") is False
                for tool in listed.tools
            )
            invoke = next(tool for tool in listed.tools if tool.name == "math.run")
            assert set(invoke.input_schema["properties"]) == {
                "operation_id",
                "inputs",
                "payload",
            }
            assert invoke.output_schema == OperationResult.model_json_schema()
            find = next(tool for tool in listed.tools if tool.name == "math.find")
            assert set(find.input_schema["properties"]) == {"request"}
            assert find.input_schema["properties"]["request"]["discriminator"] == {
                "mapping": {
                    "inspect": "#/$defs/_OperationInspectRequest",
                    "search": "#/$defs/_OperationSearchRequest",
                },
                "propertyName": "op",
            }
            assert find.output_schema["type"] == "object"
            assert find.output_schema["discriminator"] == {
                "mapping": {
                    "operation": "#/$defs/_OperationInspectionResult",
                    "discovery": "#/$defs/_OperationDiscoveryResult",
                    "error": "#/$defs/_OperationDiscoveryError",
                },
                "propertyName": "kind",
            }
            assert len(find.output_schema["oneOf"]) == 3
            assert set(
                find.output_schema["$defs"]["_OperationDiscoveryResult"]["required"]
            ) >= {"kind", "matches", "total_matches", "truncated"}
            assert set(
                find.output_schema["$defs"]["_OperationInspectionResult"]["required"]
            ) >= {"kind", "operation"}
            assert (
                find.output_schema["$defs"]["_OperationDiscoveryOperationCard"][
                    "additionalProperties"
                ]
                is False
            )
            assert (
                find.output_schema["$defs"]["_OperationDiscoveryErrorDetail"][
                    "additionalProperties"
                ]
                is False
            )
            serialized_tools = serialize_server_result(
                "tools/list",
                "2026-07-28",
                listed.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
            assert serialized_tools["tools"][0]["outputSchema"]["type"] == "object"

            with pytest.raises(MCPError) as unknown:
                await client.call_tool("math.find", {"unknown_key": "rejected"})
            assert '"code": "INVALID_INPUT"' in str(unknown.value)

            mixed_find_request = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "matrix rank",
                        "operation_id": "matrix.rank.compute",
                    }
                },
            )
            assert mixed_find_request.is_error is True

            with pytest.raises(MCPError) as retired_reasoning_input:
                await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "polynomial.expression.normalize",
                        "payload": {},
                        "reasoning_run_id": "retired",
                    },
                )
            assert '"code": "INVALID_INPUT"' in str(retired_reasoning_input.value)

            for tool_name, arguments in (
                (
                    "math.find",
                    {"request": ('{"op":"search","query":"matrix rank"}')},
                ),
                (
                    "math.run",
                    {
                        "operation_id": "polynomial.expression.normalize",
                        "payload": "{}",
                    },
                ),
            ):
                with pytest.raises(MCPError) as stringified_object:
                    await client.call_tool(tool_name, arguments)
                assert '"code": "INVALID_INPUT"' in str(stringified_object.value)

            contract_result = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": "polynomial.expression.normalize",
                    }
                },
            )
            assert isinstance(contract_result.structured_content, dict)
            contract = contract_result.structured_content
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "polynomial.expression.normalize",
                    "payload": contract["operation"]["examples"][0]["input"],
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content == OperationResult.model_validate(
                result.structured_content
            ).model_dump(mode="json")

            with pytest.raises(MCPError) as missing_resource:
                await client.read_resource("artifact://sha256/" + "f" * 64)
            assert missing_resource.value.code == -32602
            assert "requested Jacobian resource does not exist" in str(
                missing_resource.value
            )

    asyncio.run(scenario())
