from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from mcp.shared.exceptions import MCPError

from jacobian.adapters.mcp.guidance import OPERATING_GUIDE
from jacobian.adapters.mcp.server import create_server
from jacobian.contracts.capabilities import CapabilityDescriptor

CAPABILITY_TOOL_NAMES = {"capability.describe", "capability.invoke"}
MCP_TOOL_NAMES = CAPABILITY_TOOL_NAMES


def test_mcp_exposes_only_capability_tools_with_read_only_resources(
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
                "io.jacobian/core": {"version": "1"}
            }
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == MCP_TOOL_NAMES
            descriptor = json.dumps(
                {
                    "instructions": server.instructions,
                    "tools": [tool.model_dump(mode="json") for tool in listed.tools],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            assert len(descriptor) < 32_000
            assert all(
                tool.annotations is not None
                and tool.annotations.open_world_hint is False
                for tool in tools.values()
            )
            assert tools["capability.describe"].annotations is not None
            assert tools["capability.describe"].annotations.read_only_hint is True
            assert (
                "ranking is deterministic retrieval"
                in (tools["capability.describe"].description or "").lower()
            )
            describe_schema = tools["capability.describe"].input_schema
            assert all(
                tool.input_schema.get("additionalProperties") is False
                for tool in tools.values()
            )
            assert set(describe_schema["properties"]) == {
                "capability_id",
                "query",
                "domain",
                "mode",
                "input_kind",
                "artifact_type",
                "limit",
                "cursor",
                "view",
            }
            assert set(tools["capability.invoke"].input_schema["properties"]) == {
                "capability_id",
                "payload",
                "mode",
            }
            with pytest.raises(MCPError) as unknown_argument:
                await client.call_tool(
                    "capability.describe",
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
            assert "Search any outcomes or concepts" in rendered_prompt

            discovery_result = await client.call_tool(
                "capability.describe",
                {"query": "search mathematical knowledge", "limit": 3},
            )
            discovery = json.loads(discovery_result.content[0].text)
            assert discovery["kind"] == "discovery"
            assert 0 < len(discovery["matches"]) <= 3
            assert "input_schema" not in discovery["matches"][0]
            assert discovery["next_step"] == {
                "tool": "capability.describe",
                "argument": "capability_id",
                "choose_from": "matches[].capability_id",
            }
            assert discovery["routing_guidance"]["inspect_candidates"] == (
                "Inspect only the strongest one or two domain-relevant matches; "
                "search again only when none fits the required outcome."
            )
            assert (
                "producer result"
                in discovery["routing_guidance"]["verification_handoff"]
            )

            catalog_result = await client.read_resource("capability://catalog")
            catalog = json.loads(catalog_result.contents[0].text)
            capability_ids = {
                descriptor["capability_id"] for descriptor in catalog["capabilities"]
            }
            assert "knowledge.search" in capability_ids
            assert all(
                descriptor["provider_runtime"]["availability"] == "AVAILABLE"
                for descriptor in catalog["capabilities"]
            )
            if "lean.check" in capability_ids:
                lean_result = await client.call_tool(
                    "capability.describe",
                    {"capability_id": "lean.check", "view": "FULL"},
                )
                lean_contract = json.loads(lean_result.content[0].text)
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


def test_mcp_describes_and_invokes_capabilities(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            described = await client.call_tool(
                "capability.describe",
                {"capability_id": "knowledge.search", "view": "CONTRACT"},
            )
            contract = json.loads(described.content[0].text)
            assert contract["view"] == "CONTRACT"
            assert contract["capability"]["capability_id"] == "knowledge.search"
            assert contract["capability"]["provider_runtime"]["digest"].startswith(
                "sha256:"
            )
            assert "configuration" not in contract["capability"]["provider_runtime"]
            assert "output_schema" not in contract["capability"]
            assert "output_schema_summary" in contract["capability"]

            result = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "knowledge.search",
                    "mode": "EXPLORE",
                    "payload": {"query": "counterexample", "limit": 5},
                },
            )
            inline_result = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert isinstance(inline_result.structured_content, dict)
            assert inline_result.structured_content["episode_uri"] is None
            assert inline_result.structured_content["artifact_uris"] == []
            response = json.loads(result.content[0].text)
            assert response["execution"]["status"] == "COMPLETED"
            assert response["assurance"]["level"] == "COMPUTED"
            assert isinstance(result.structured_content, dict)
            assert "mcp_projection" not in result.structured_content
            assert result.structured_content["output"] == response["output"]
            runtime = contract["capability"]["provider_runtime"]
            assert response["provider"] == contract["capability"]["provider"]
            assert response["provider_digest"] == runtime["digest"]

            matching_description = await client.call_tool(
                "capability.describe",
                {
                    "capability_id": ("graph.invariant.maximum_matching.compute"),
                    "view": "CONTRACT",
                },
            )
            matching_contract = json.loads(matching_description.content[0].text)
            assert matching_contract["capability"]["version"] == "2"
            assert matching_contract["invocations"][0]["name"] == ("triangle_with_tail")
            assert matching_contract["related_capabilities"] == [
                {
                    "capability_id": ("graph.invariant.maximum_matching.verify"),
                    "relationship": (
                        "independently replay the stored Tutte-Berge certificate"
                    ),
                }
            ]

            unknown = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "missing.capability",
                    "mode": "EXPLORE",
                    "payload": {},
                },
            )
            unknown_result = json.loads(unknown.content[0].text)
            assert unknown.is_error is False
            assert unknown_result["execution"]["status"] == "ERROR"
            assert unknown_result["output"]["error"]["code"] == "UNKNOWN_CAPABILITY"
            assert unknown_result["assurance"]["level"] != "VERIFIED"

    asyncio.run(scenario())


def test_mcp_inline_results_do_not_emit_resource_links(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content["episode_uri"] is None
            assert result.structured_content["artifact_uris"] == []
            assert [
                block for block in result.content if block.type == "resource_link"
            ] == []

    asyncio.run(scenario())


def test_mcp_exact_description_layers_summary_contract_and_full_views(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            summary_result = await client.call_tool(
                "capability.describe",
                {"capability_id": "polynomial.expression.normalize"},
            )
            contract_result = await client.call_tool(
                "capability.describe",
                {
                    "capability_id": "polynomial.expression.normalize",
                    "view": "CONTRACT",
                },
            )
            full_result = await client.call_tool(
                "capability.describe",
                {
                    "capability_id": "polynomial.expression.normalize",
                    "view": "FULL",
                },
            )
            summary = json.loads(summary_result.content[0].text)
            contract = json.loads(contract_result.content[0].text)
            full = json.loads(full_result.content[0].text)

            assert summary["view"] == "SUMMARY"
            assert "input_schema" not in summary["capability"]
            assert summary["capability"]["input_schema_summary"]["type"] == "object"
            assert summary["capability"]["has_invocation_examples"] is True
            assert summary["capability"]["accepted_input_kinds"] == [
                "STRUCTURED_REQUEST"
            ]
            assert summary["capability"]["accepted_artifact_types"] == []
            assert "invocations" not in summary
            assert "CONTRACT" in summary["next_views"]
            assert "all-orders" in summary["scope_rule"]["bounded_repetition"]
            assert contract["view"] == "CONTRACT"
            assert contract["capability"]["input_schema"]["type"] == "object"
            assert contract["capability"]["accepted_input_kinds"] == [
                "STRUCTURED_REQUEST"
            ]
            assert contract["capability"]["accepted_artifact_types"] == []
            assert contract["invocations"]
            assert full["view"] == "FULL"
            assert "output_schema" in full["capability"]
            assert "configuration" in full["capability"]["provider_runtime"]
            CapabilityDescriptor.model_validate(full["capability"])
            payload = contract["invocations"][0]["arguments"]["payload"]
            contract_validator = Draft202012Validator(
                contract["capability"]["input_schema"]
            )
            full_validator = Draft202012Validator(full["capability"]["input_schema"])
            assert contract_validator.is_valid(payload)
            assert full_validator.is_valid(payload)
            assert not contract_validator.is_valid({})
            assert not full_validator.is_valid({})
            assert len(summary_result.content[0].text) * 100 < (
                len(contract_result.content[0].text) * 40
            )
            assert len(contract_result.content[0].text) * 100 < (
                len(full_result.content[0].text) * 51
            )

    asyncio.run(scenario())
