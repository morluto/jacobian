from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError

from jacobian.adapters.mcp.guidance import OPERATING_GUIDE
from jacobian.adapters.mcp.server import create_server

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


def test_mcp_exposes_only_math_tools_with_read_only_resources(
    tmp_path: Path,
) -> None:
    server = create_server(tmp_path)
    assert server.instructions is not None
    assert "assurance level VERIFIED" in server.instructions

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
            assert tools["math.run"].annotations.destructive_hint is True
            assert (
                "ranking is deterministic lexical retrieval"
                in (tools["math.find"].description or "").lower()
            )
            describe_schema = tools["math.find"].input_schema
            assert all(
                tool.input_schema.get("additionalProperties") is False
                for tool in tools.values()
            )
            assert set(describe_schema["properties"]) == {
                "capability_id",
                "query",
                "domain",
                "input_kind",
                "artifact_type",
                "limit",
                "cursor",
                "view",
            }
            assert set(tools["math.run"].input_schema["properties"]) == {
                "capability_id",
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
                    "jacobian-instructions",
                    "jacobian://instructions",
                    "text/markdown",
                ),
                (
                    "capability-catalog",
                    "capability://catalog",
                    "application/json",
                ),
                (
                    "reference-catalog",
                    "reference://catalog",
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
                (
                    "experiment",
                    "experiment://{experiment_id}",
                    "application/json",
                ),
                (
                    "experiment-accounting",
                    "experiment://{experiment_id}/accounting",
                    "application/json",
                ),
                (
                    "experiment-scope",
                    "experiment://{experiment_id}/scope",
                    "application/json",
                ),
                (
                    "experiment-archive",
                    "experiment://{experiment_id}/archive",
                    "application/json",
                ),
            }
            instructions = await client.read_resource("jacobian://instructions")
            assert instructions.contents[0].text == OPERATING_GUIDE

            prompts = await client.list_prompts()
            prompt_names = {prompt.name for prompt in prompts.prompts}
            assert prompt_names == {
                "jacobian-check-evidence",
                "jacobian-discover",
            }
            discovery_prompt = await client.get_prompt(
                "jacobian-discover",
                {"task": "Explore structures related to a conjecture."},
            )
            rendered_prompt = discovery_prompt.messages[0].content.text
            assert "research strategy" in rendered_prompt
            assert "desired local mathematical outcome" in rendered_prompt
            assert "Available affordances" in rendered_prompt

            discovery_result = await client.call_tool(
                "math.find",
                {"query": "search mathematical knowledge", "limit": 3},
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
            assert "output_schema_summary" in operation_card
            assert operation_card["scope"] == "EXACT_SUPPLIED_INPUT_OR_CLAIM"
            assert operation_card["provider_availability"] == "AVAILABLE"
            assert isinstance(operation_card["related_capabilities"], list)

            absent_result = await client.call_tool(
                "math.find",
                {"query": "quuxonium frobnicator", "domain": "polynomial"},
            )
            assert isinstance(absent_result.structured_content, dict)
            absent = absent_result.structured_content
            assert absent["portfolio_fit"] == "NO_LEXICAL_MATCHES"
            assert absent["matches"] == []
            recovery_actions = {
                option["action"] for option in absent["available_recovery_paths"]
            }
            assert recovery_actions == {
                "reformulate_query",
                "remove_filters",
                "browse",
                "inspect_catalog",
            }
            browse = next(
                option
                for option in absent["available_recovery_paths"]
                if option["action"] == "browse"
            )
            assert browse == {
                "action": "browse",
                "tool": "math.find",
                "arguments": {},
            }

            unknown_domain_result = await client.call_tool(
                "math.find",
                {
                    "query": "compute exact event probability",
                    "domain": "arithmetic",
                },
            )
            assert isinstance(unknown_domain_result.structured_content, dict)
            unknown_domain = unknown_domain_result.structured_content
            assert unknown_domain["domain_filter_status"] == "UNKNOWN"
            assert (
                "matches no installed capability"
                in unknown_domain["domain_filter_basis"]
            )
            assert (
                "lexical fit outside that filter was not assessed"
                in (unknown_domain["portfolio_fit_basis"])
            )
            assert unknown_domain["recovery_paths_are_unranked"] is True
            assert {
                "action": "remove_unknown_domain_filter",
                "tool": "math.find",
                "rejected_domain": "arithmetic",
                "change": "Retry without the unrecognized domain filter.",
            } in unknown_domain["available_recovery_paths"]
            assert {
                "action": "reformulate_query",
                "tool": "math.find",
                "change": "Use different or broader mathematical language for query.",
            } in unknown_domain["available_recovery_paths"]

            browse_unknown_result = await client.call_tool(
                "math.find",
                {"domain": "arithmetic"},
            )
            assert isinstance(browse_unknown_result.structured_content, dict)
            browse_unknown = browse_unknown_result.structured_content
            assert browse_unknown["domain_filter_status"] == "UNKNOWN"
            assert browse_unknown["portfolio_fit"] == "UNFILTERED"
            assert browse_unknown["routing_status"] == "UNFILTERED"
            assert browse_unknown["recovery_paths_are_unranked"] is True
            assert {
                "action": "remove_unknown_domain_filter",
                "tool": "math.find",
                "rejected_domain": "arithmetic",
                "change": "Retry without the unrecognized domain filter.",
            } in browse_unknown["available_recovery_paths"]
            assert all(
                option["action"] != "reformulate_query"
                for option in browse_unknown["available_recovery_paths"]
            )

            hidden_domain_result = await client.call_tool(
                "math.find",
                {"domain": "artifact"},
            )
            assert isinstance(hidden_domain_result.structured_content, dict)
            hidden_domain = hidden_domain_result.structured_content
            assert hidden_domain["domain_filter_status"] == "MATCHED"
            assert hidden_domain["matches"] == []
            assert "artifact.put" not in {
                match["capability_id"] for match in hidden_domain["matches"]
            }

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
                    {"capability_id": "lean.check", "view": "FULL"},
                )
                assert isinstance(lean_result.structured_content, dict)
                lean_contract = lean_result.structured_content
                assert lean_contract["view"] == "FULL"
                lean_runtime = lean_contract["capability"]["provider_runtime"]
                assert lean_runtime["install_tier"] == "T3"
                assert (
                    lean_runtime["configuration"]["profiles"]["MATHLIB"][
                        "mathlib_commit"
                    ]
                    == "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
                )
                assert "runtime" not in lean_contract

            reference_result = await client.read_resource("reference://catalog")
            references = json.loads(reference_result.contents[0].text)
            assert references["matrices"]["plugin_id"].startswith("artifact://sha256/")

    asyncio.run(scenario())
