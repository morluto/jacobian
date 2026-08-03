"""Executable conformance checks for the pinned MCP Python SDK boundary."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from pathlib import Path

import pytest
from mcp.server.extension import Extension, ResourceBinding, ToolBinding

from jacobian.adapters.mcp.server import JacobianCoreExtension, create_server
from jacobian.contracts.capabilities import CapabilityResult


def test_mcp_sdk_is_exactly_pinned_and_v2_bindings_are_used() -> None:
    assert importlib.metadata.version("mcp") == "2.0.0"
    assert importlib.metadata.version("mcp-types") == "2.0.0"

    extension = JacobianCoreExtension(None, None)
    assert isinstance(extension, Extension)
    assert extension.identifier == "io.jacobian/core"
    assert extension.settings() == {"version": "1"}
    assert all(isinstance(binding, ToolBinding) for binding in extension.tools())
    assert all(
        isinstance(binding, ResourceBinding) for binding in extension.resources()
    )
    assert tuple(binding.kwargs["name"] for binding in extension.tools()) == (
        "capability.describe",
        "capability.invoke",
    )


def test_mcp_v2_static_validation_context_errors_and_structured_resources(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client
        from mcp.shared.exceptions import MCPError

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            listed = await client.list_tools()
            assert all(
                tool.input_schema.get("additionalProperties") is False
                for tool in listed.tools
            )
            invoke = next(
                tool for tool in listed.tools if tool.name == "capability.invoke"
            )
            assert invoke.output_schema == CapabilityResult.model_json_schema()

            with pytest.raises(MCPError) as unknown:
                await client.call_tool(
                    "capability.describe", {"unknown_key": "rejected"}
                )
            assert '"code": "INVALID_INPUT"' in str(unknown.value)

            contract = json.loads(
                (
                    await client.call_tool(
                        "capability.describe",
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
                "capability.invoke",
                {
                    "capability_id": "polynomial.expression.normalize",
                    "mode": "EXPLORE",
                    "payload": contract["invocations"][0]["arguments"]["payload"],
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content == CapabilityResult.model_validate(
                result.structured_content
            ).model_dump(mode="json")
            episode_uri = result.structured_content["episode_uri"]
            assert isinstance(episode_uri, str)
            resource = await client.read_resource(episode_uri)
            assert json.loads(resource.contents[0].text)["artifact_uri"] == episode_uri

            with pytest.raises(MCPError):
                await client.read_resource("artifact://sha256/" + "f" * 64)

    asyncio.run(scenario())
