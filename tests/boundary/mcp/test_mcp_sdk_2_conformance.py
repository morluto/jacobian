"""Executable conformance checks for the pinned MCP Python SDK boundary."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from pathlib import Path

import pytest
from mcp.server.extension import Extension, ResourceBinding, ToolBinding
from mcp_types.methods import serialize_server_result

import jacobian.adapters.mcp.server as server_module
from jacobian.adapters.mcp.server import JacobianCoreExtension, create_server
from jacobian.contracts.capabilities import CapabilityResult


def test_mcp_sdk_is_exactly_pinned_and_v2_bindings_are_used() -> None:
    assert importlib.metadata.version("mcp") == "2.0.0"
    assert importlib.metadata.version("mcp-types") == "2.0.0"

    extension = JacobianCoreExtension(None, None)
    assert isinstance(extension, Extension)
    assert extension.identifier == "io.jacobian/core"
    assert extension.settings() == {"version": "2"}
    assert all(isinstance(binding, ToolBinding) for binding in extension.tools())
    assert all(
        isinstance(binding, ResourceBinding) for binding in extension.resources()
    )
    assert tuple(binding.kwargs["name"] for binding in extension.tools()) == (
        "math.find",
        "math.run",
    )


def test_mcp_v2_static_validation_context_errors_and_structured_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(server_module, "Context", raising=False)

    async def scenario() -> None:
        from mcp import Client
        from mcp.shared.exceptions import MCPError

        server = create_server(tmp_path)
        assert not hasattr(server_module, "Context")
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            assert all(
                tool.input_schema.get("additionalProperties") is False
                for tool in listed.tools
            )
            invoke = next(tool for tool in listed.tools if tool.name == "math.run")
            assert set(invoke.input_schema["properties"]) == {
                "capability_id",
                "payload",
            }
            assert invoke.output_schema == CapabilityResult.model_json_schema()
            find = next(tool for tool in listed.tools if tool.name == "math.find")
            assert find.output_schema["type"] == "object"
            assert find.output_schema["discriminator"] == {
                "mapping": {
                    "capability": "#/$defs/_CapabilityInspectionResult",
                    "discovery": "#/$defs/_CapabilityDiscoveryResult",
                    "error": "#/$defs/_CapabilityDiscoveryError",
                },
                "propertyName": "kind",
            }
            assert len(find.output_schema["oneOf"]) == 3
            assert set(
                find.output_schema["$defs"]["_CapabilityDiscoveryResult"]["required"]
            ) >= {"kind", "matches", "total_matches", "truncated"}
            assert set(
                find.output_schema["$defs"]["_CapabilityInspectionResult"]["required"]
            ) >= {"kind", "view", "capability"}
            assert (
                find.output_schema["$defs"]["_CapabilityDiscoveryOperationCard"][
                    "additionalProperties"
                ]
                is False
            )
            assert (
                find.output_schema["$defs"]["_CapabilityDiscoveryErrorDetail"][
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

            with pytest.raises(MCPError) as retired_reasoning_input:
                await client.call_tool(
                    "math.run",
                    {
                        "capability_id": "polynomial.expression.normalize",
                        "payload": {},
                        "reasoning_run_id": "retired",
                    },
                )
            assert '"code": "INVALID_INPUT"' in str(retired_reasoning_input.value)

            invalid_discovery_view = await client.call_tool(
                "math.find",
                {
                    "query": "normalize a polynomial expression",
                    "view": "FULL",
                },
            )
            assert invalid_discovery_view.is_error is True
            invalid_view_error = json.loads(invalid_discovery_view.content[0].text)
            assert invalid_view_error["error"]["code"] == "INVALID_INPUT"
            assert invalid_view_error["error"]["stage"] == "math.find"

            contract = json.loads(
                (
                    await client.call_tool(
                        "math.find",
                        {
                            "capability_id": "polynomial.expression.normalize",
                            "view": "CONTRACT",
                        },
                    )
                )
                .content[0]
                .text
            )
            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "polynomial.expression.normalize",
                    "payload": contract["invocations"][0]["arguments"]["payload"],
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content == CapabilityResult.model_validate(
                result.structured_content
            ).model_dump(mode="json")

            with pytest.raises(MCPError) as missing_resource:
                await client.read_resource("artifact://sha256/" + "f" * 64)
            assert missing_resource.value.code == -32602
            assert "requested Jacobian resource does not exist" in str(
                missing_resource.value
            )

    asyncio.run(scenario())
