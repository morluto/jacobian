from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError

from jacobian.adapters.mcp.deployment_identity import DeploymentIdentity
from jacobian.adapters.mcp.server import create_server
from jacobian.runtime import CheckerAuthorityMode
from tests.boundary.mcp.mcp_support import open_focused_mcp_server

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


def test_managed_server_advertises_immutable_deployment_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = DeploymentIdentity(
        revision="a" * 40,
        package_version=version("jacobian"),
    )
    monkeypatch.setattr(
        "jacobian.adapters.mcp.server.load_deployment_identity",
        lambda: identity,
    )

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            resources = await client.list_resources()
            assert {
                (resource.name, str(resource.uri)) for resource in resources.resources
            } == {
                ("capability-catalog", "capability://catalog"),
                ("deployment-identity", "deployment://identity"),
            }
            result = await client.read_resource("deployment://identity")
            assert json.loads(result.contents[0].text) == identity.model_dump(
                mode="json"
            )

    with open_focused_mcp_server(tmp_path) as server:
        asyncio.run(scenario())


def test_mcp_exposes_only_math_tools_with_read_only_resources(
    tmp_path: Path,
) -> None:
    server = create_server(tmp_path, checker_authority=CheckerAuthorityMode.NONE)
    assert server.instructions is not None
    assert "local verification record URI" in server.instructions

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            assert client.instructions == server.instructions
            assert client.server_info.version == version("jacobian")
            assert client.server_capabilities.extensions == {
                "io.jacobian/core": {"version": "2"}
            }
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == MCP_TOOL_NAMES
            assert all(
                tool.annotations is not None
                and tool.annotations.open_world_hint is False
                for tool in tools.values()
            )
            assert tools["math.find"].annotations is not None
            assert tools["math.find"].annotations.read_only_hint is True
            assert tools["math.run"].annotations is not None
            assert tools["math.run"].annotations.destructive_hint is False
            assert tools["math.run"].annotations.read_only_hint is False
            assert tools["math.run"].annotations.idempotent_hint is False
            assert (
                "ranking is deterministic lexical retrieval"
                in (tools["math.find"].description or "").lower()
            )
            describe_schema = tools["math.find"].input_schema
            assert all(
                tool.input_schema.get("additionalProperties") is False
                for tool in tools.values()
            )
            assert set(describe_schema["properties"]) == {"request"}
            request_schema = describe_schema["properties"]["request"]
            assert request_schema["discriminator"]["propertyName"] == "op"
            assert set(tools["math.run"].input_schema["properties"]) == {
                "capability_id",
                "inputs",
                "payload",
            }
            with pytest.raises(MCPError) as unknown_argument:
                await client.call_tool(
                    "math.find",
                    {"capabilty_id": "polynomial.compute.gcd"},
                )
            assert '"code": "INVALID_INPUT"' in str(unknown_argument.value)
            resources = await client.list_resources()
            resource_inventory = {
                (resource.name, str(resource.uri), resource.mime_type)
                for resource in resources.resources
            }
            assert resource_inventory == {
                (
                    "capability-catalog",
                    "capability://catalog",
                    "application/json",
                ),
            }

            templates = await client.list_resource_templates()
            template_inventory = {
                (template.name, template.uri_template, template.mime_type)
                for template in templates.resource_templates
            }
            assert template_inventory == {
                (
                    "artifact",
                    "artifact://sha256/{digest}",
                    "application/json",
                ),
            }
            prompts = await client.list_prompts()
            assert prompts.prompts == []

            discovery_result = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "search mathematical knowledge",
                        "limit": 3,
                    }
                },
            )
            assert isinstance(discovery_result.structured_content, dict)
            discovery = discovery_result.structured_content
            assert discovery["kind"] == "discovery"
            assert 0 < len(discovery["matches"]) <= 3
            assert "input_schema" not in discovery["matches"][0]
            assert "next_step" not in discovery
            assert "routing_guidance" not in discovery
            operation_card = discovery["matches"][0]
            assert operation_card["accepted_input_kinds"]
            assert operation_card["provider_availability"] == "AVAILABLE"
            assert "input_schema" not in operation_card

            absent_result = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "quuxonium frobnicator",
                        "domain": "polynomial",
                    }
                },
            )
            assert isinstance(absent_result.structured_content, dict)
            absent = absent_result.structured_content
            assert absent["matches"] == []
            assert absent["catalog_resource"] == "capability://catalog"

            unknown_domain_result = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "search",
                        "query": "compute exact event probability",
                        "domain": "arithmetic",
                    }
                },
            )
            assert isinstance(unknown_domain_result.structured_content, dict)
            unknown_domain = unknown_domain_result.structured_content
            assert unknown_domain["domain"] == "arithmetic"
            assert unknown_domain["matches"] == []
            assert unknown_domain["catalog_resource"] == "capability://catalog"

            catalog_result = await client.read_resource("capability://catalog")
            catalog = json.loads(catalog_result.contents[0].text)
            capability_ids = {
                descriptor["capability_id"] for descriptor in catalog["capabilities"]
            }
            assert all(
                descriptor["provider_runtime"]["availability"] == "AVAILABLE"
                for descriptor in catalog["capabilities"]
            )
            if "lean.check" in capability_ids:
                lean_result = await client.call_tool(
                    "math.find",
                    {
                        "request": {
                            "op": "inspect",
                            "capability_id": "lean.check",
                        }
                    },
                )
                assert isinstance(lean_result.structured_content, dict)
                lean_contract = lean_result.structured_content
                lean_runtime = lean_contract["capability"]["provider_runtime"]
                assert lean_runtime["install_tier"] == "T3"
                assert (
                    lean_runtime["configuration"]["profiles"]["MATHLIB"][
                        "mathlib_commit"
                    ]
                    == "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
                )
                assert "runtime" not in lean_contract

    asyncio.run(scenario())
