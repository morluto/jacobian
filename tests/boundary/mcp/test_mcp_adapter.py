from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from jacobian.adapters.mcp.guidance import OPERATING_GUIDE
from jacobian.adapters.mcp.projections import WORKSPACE_TOOL_NAMES
from jacobian.adapters.mcp.server import create_server
from jacobian.capabilities import CapabilityPolicy
from jacobian.contracts.capabilities import CapabilityDescriptor

CAPABILITY_TOOL_NAMES = {"capability.describe", "capability.invoke"}
MCP_TOOL_NAMES = CAPABILITY_TOOL_NAMES | WORKSPACE_TOOL_NAMES


def test_mcp_exposes_capability_and_workspace_tools_with_read_only_resources(
    tmp_path: Path,
) -> None:
    server = create_server(tmp_path)
    assert server.instructions is not None
    assert "assurance level VERIFIED" in server.instructions
    assert "workspace entry never promotes" in server.instructions

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
                "view",
            }
            unknown_argument = await client.call_tool(
                "capability.describe",
                {"capabilty_id": "polynomial.compute.gcd"},
            )
            assert unknown_argument.is_error is True
            assert '"code": "INVALID_INPUT"' in unknown_argument.content[0].text
            assert tools["workspace.open"].annotations is not None
            assert tools["workspace.open"].annotations.idempotent_hint is True
            assert tools["workspace.write"].annotations is not None
            assert tools["workspace.write"].annotations.idempotent_hint is True
            assert tools["workspace.query"].annotations is not None
            assert tools["workspace.query"].annotations.read_only_hint is True
            assert all(
                tools[name].output_schema is None for name in WORKSPACE_TOOL_NAMES
            )

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


def test_internal_projection_strategies_are_composable_not_model_parameters(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        for strategy in (
            "FULL_INLINE",
            "COMPACT_URI_TEXT",
            "COMPACT_URI_TEXT_RESOURCE_LINK",
        ):
            server = create_server(
                tmp_path / strategy,
                _projection_strategy=strategy,
            )
            async with Client(server, raise_exceptions=True) as client:
                listed = await client.list_tools()
                invoke_schema = next(
                    tool.input_schema
                    for tool in listed.tools
                    if tool.name == "capability.invoke"
                )
                assert "projection_strategy" not in json.dumps(invoke_schema)
                result = await client.call_tool(
                    "capability.invoke",
                    {
                        "capability_id": "integer.compute.gcd",
                        "mode": "EXPLORE",
                        "payload": {"left": "84", "right": "30"},
                    },
                )
                assert isinstance(result.structured_content, dict)
                links = [
                    block for block in result.content if block.type == "resource_link"
                ]
                if strategy == "FULL_INLINE":
                    assert len(links) == 0
                    assert json.loads(result.content[0].text)["output"]
                elif strategy == "COMPACT_URI_TEXT":
                    assert len(links) == 0
                else:
                    assert len(links) == 1
                assert result.structured_content["output"]

    asyncio.run(scenario())


def test_mcp_workspace_schema_and_fail_closed_round_trip(
    tmp_path: Path,
) -> None:
    server = create_server(tmp_path)

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            open_schema = tools["workspace.open"].input_schema
            write_tool = tools["workspace.write"]
            write_schema = write_tool.input_schema
            query_schema = tools["workspace.query"].input_schema

            assert open_schema["properties"]["idempotency_key"]["pattern"] == (
                "^[A-Za-z0-9._:-]{8,128}$"
            )
            assert open_schema["properties"]["name"]["maxLength"] == 128
            assert open_schema["properties"]["problem"]["maxLength"] == 16_384
            assert open_schema["properties"]["tags"]["anyOf"][0]["maxItems"] == 16
            assert write_tool.description is not None
            assert "base_revision (never revision_id)" in write_tool.description
            assert "never a batch wrapper" in write_tool.description
            assert "client_ref, never ref" in write_tool.description
            assert "never depends_on_refs" in write_tool.description
            write_properties = write_schema["properties"]
            assert write_properties["workspace_id"]["pattern"] == (
                "^workspace://[0-9a-f]{32}$"
            )
            assert write_properties["branch_id"]["pattern"] == (
                "^branch://[0-9a-f]{32}$"
            )
            assert write_properties["base_revision"]["pattern"] == (
                "^revision://[0-9a-f]{32}$"
            )
            for field_name in ("scratch", "findings", "attempts", "marks"):
                assert write_properties[field_name]["anyOf"][0]["maxItems"] == 64
            assert query_schema["properties"]["limit"]["minimum"] == 1
            assert query_schema["properties"]["limit"]["maximum"] == 50
            assert query_schema["properties"]["workspace_id"]["pattern"] == (
                "^workspace://[0-9a-f]{32}$"
            )
            assert (
                query_schema["properties"]["target_card_id"]["anyOf"][0]["pattern"]
                == "^card://[0-9a-f]{32}$"
            )

            finding_kinds = write_schema["$defs"]["WorkspaceFindingKind"]["enum"]
            attempt_outcomes = write_schema["$defs"]["WorkspaceAttemptOutcome"]["enum"]
            assert "GOAL" in finding_kinds
            assert "OPEN_GOAL" not in finding_kinds
            assert "COMPLETED" in attempt_outcomes
            assert "SUCCEEDED" not in attempt_outcomes
            mark_schema = write_schema["$defs"]["WorkspaceMarkDraft"]
            assert "reason" in mark_schema["properties"]
            assert "summary" not in mark_schema["properties"]

            canonical_payload = {
                "workspace_id": "workspace://" + ("0" * 32),
                "branch_id": "branch://" + ("0" * 32),
                "base_revision": "revision://" + ("0" * 32),
                "idempotency_key": "schema-canonical-001",
                "findings": [
                    {
                        "client_ref": "G1",
                        "kind": "GOAL",
                        "title": "Open work",
                        "body": "Canonical unfinished work.",
                    }
                ],
                "attempts": [
                    {
                        "client_ref": "T1",
                        "target_ref": "G1",
                        "method": "direct",
                        "outcome": "COMPLETED",
                        "summary": "The operational attempt completed.",
                    }
                ],
                "marks": [
                    {
                        "client_ref": "M1",
                        "target_ref": "G1",
                        "state": "CLOSED",
                        "reason": "The goal was explicitly closed.",
                    }
                ],
            }
            write_validator = Draft202012Validator(write_schema)
            assert write_validator.is_valid(canonical_payload), list(
                write_validator.iter_errors(canonical_payload)
            )

            misdirected_result = await client.call_tool(
                "capability.describe",
                {"capability_id": "workspace.write"},
            )
            misdirected = json.loads(misdirected_result.content[0].text)
            assert misdirected["error"]["code"] == "UNKNOWN_CAPABILITY"
            assert "direct MCP tool" in misdirected["error"]["hint"]

            opened_result = await client.call_tool(
                "workspace.open",
                {
                    "idempotency_key": "mcp-workspace-open-001",
                    "name": "MCP workspace",
                    "problem": "Record a goal and one completed attempt.",
                },
            )
            opened = json.loads(opened_result.content[0].text)

            rejected_result = await client.call_tool(
                "workspace.write",
                {
                    "idempotency_key": "mcp-workspace-unknown-field-001",
                    "workspace_id": opened["workspace_id"],
                    "branch_id": opened["branch_id"],
                    "base_revision": opened["revision_id"],
                    "cards": [],
                    "attempts": [
                        {
                            "client_ref": "T0",
                            "target_ref": opened["problem_card_id"],
                            "method": "must-not-commit",
                            "outcome": "COMPLETED",
                            "summary": "Unknown input rejects the entire write.",
                        }
                    ],
                },
            )
            assert rejected_result.is_error is True
            assert '"code": "INVALID_INPUT"' in rejected_result.content[0].text

            second_problem_result = await client.call_tool(
                "workspace.write",
                {
                    "idempotency_key": "mcp-workspace-second-problem-001",
                    "workspace_id": opened["workspace_id"],
                    "branch_id": opened["branch_id"],
                    "base_revision": opened["revision_id"],
                    "findings": [
                        {
                            "client_ref": "P2",
                            "kind": "PROBLEM",
                            "title": "Hidden second problem",
                            "body": "Only workspace.open creates the problem.",
                        }
                    ],
                },
            )
            assert second_problem_result.is_error is True
            assert '"code": "INVALID_INPUT"' in second_problem_result.content[0].text

            unchanged_result = await client.call_tool(
                "workspace.query",
                {
                    "workspace_id": opened["workspace_id"],
                    "branch_id": opened["branch_id"],
                    "revision_id": opened["revision_id"],
                    "view": "RESUME",
                },
            )
            unchanged = json.loads(unchanged_result.content[0].text)
            assert unchanged["revision_id"] == opened["revision_id"]
            assert unchanged["resume"]["recent_attempts"] == []

            write_result = await client.call_tool(
                "workspace.write",
                {
                    "idempotency_key": "mcp-workspace-write-001",
                    "workspace_id": opened["workspace_id"],
                    "branch_id": opened["branch_id"],
                    "base_revision": opened["revision_id"],
                    "findings": [
                        {
                            "client_ref": "A1",
                            "kind": "ASSUMPTION",
                            "title": "Temporary scope",
                            "body": "Assume a temporary finite scope.",
                        },
                        {
                            "client_ref": "G1",
                            "kind": "GOAL",
                            "title": "MCP goal",
                            "body": "Close the remaining case.",
                            "assumption_refs": ["A1"],
                        },
                    ],
                    "attempts": [
                        {
                            "client_ref": "T1",
                            "target_ref": "G1",
                            "method": "direct",
                            "outcome": "COMPLETED",
                            "summary": "The operational attempt completed.",
                        }
                    ],
                    "focus": {"active_ref": "G1", "pinned_refs": ["G1"]},
                },
            )
            written = json.loads(write_result.content[0].text)
            assert written["findings_written"] == 2
            assert written["attempts_written"] == 1
            assert len(written["unverified_finding_ids"]) == 2
            assert (
                "cannot establish an exact mathematical conclusion"
                in written["assurance_notice"]
            )

            mark_result = await client.call_tool(
                "workspace.write",
                {
                    "idempotency_key": "mcp-workspace-mark-001",
                    "workspace_id": opened["workspace_id"],
                    "branch_id": opened["branch_id"],
                    "base_revision": written["revision_id"],
                    "marks": [
                        {
                            "client_ref": "M1",
                            "target_ref": written["id_map"]["A1"],
                            "state": "RETRACTED",
                            "reason": "The temporary scope was withdrawn.",
                        }
                    ],
                },
            )
            marked = json.loads(mark_result.content[0].text)
            assert marked["marks_written"] == 1

            context_result = await client.call_tool(
                "workspace.query",
                {
                    "workspace_id": opened["workspace_id"],
                    "branch_id": opened["branch_id"],
                    "revision_id": marked["revision_id"],
                    "view": "CONTEXT",
                    "target_card_id": written["id_map"]["G1"],
                },
            )
            context = json.loads(context_result.content[0].text)["context"]
            assert context["target"]["kind"] == "GOAL"
            assert context["target"]["stale"] is True
            assert context["target"]["stale_due_to_ids"] == [written["id_map"]["A1"]]
            assert context["target"]["verification"] == "UNVERIFIED"
            assert context["dependencies"][0]["state"] == "RETRACTED"
            assert context["recent_attempts"][0]["outcome"] == "COMPLETED"
            assert context["recent_attempts"][0]["verification"] == "UNVERIFIED"

    asyncio.run(scenario())


def test_mcp_workspace_write_treats_explicit_null_collections_as_empty(
    tmp_path: Path,
) -> None:
    server = create_server(tmp_path)

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            opened_result = await client.call_tool(
                "workspace.open",
                {
                    "idempotency_key": "mcp-workspace-open-null-001",
                    "name": "Nullable collections",
                    "problem": "Explicit null and omission have the same meaning.",
                },
            )
            opened = json.loads(opened_result.content[0].text)
            written_result = await client.call_tool(
                "workspace.write",
                {
                    "idempotency_key": "mcp-workspace-write-null-001",
                    "workspace_id": opened["workspace_id"],
                    "branch_id": opened["branch_id"],
                    "base_revision": opened["revision_id"],
                    "scratch": None,
                    "findings": [
                        {
                            "client_ref": "G1",
                            "kind": "GOAL",
                            "title": "One real mutation",
                            "body": "The other nullable collections are empty.",
                        }
                    ],
                    "attempts": None,
                    "marks": None,
                    "focus": None,
                },
            )

            assert written_result.is_error is False
            written = json.loads(written_result.content[0].text)
            assert written["scratch_written"] == 0
            assert written["findings_written"] == 1
            assert written["attempts_written"] == 0
            assert written["marks_written"] == 0
            assert written["focus_updated"] is False

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
            durable_result = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            result_link = next(
                block
                for block in durable_result.content
                if block.type == "resource_link"
            )
            durable = await client.read_resource(result_link.uri)
            durable_payload = json.loads(durable.contents[0].text)
            assert durable_payload["artifact_uri"] == result_link.uri
            assert durable_payload["manifest"]["object_digest"].startswith("sha256:")
            assert durable_payload["payload"]["capability_id"] == "integer.compute.gcd"
            response = json.loads(result.content[0].text)
            assert response["execution"]["status"] == "COMPLETED"
            assert response["assurance"]["level"] == "COMPUTED"
            assert response["mcp_projection"]["view"] == "STANDARD"
            assert response["mcp_projection"]["output_complete"] is True
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

            summary_result = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "view": "SUMMARY",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            summary = json.loads(summary_result.content[0].text)
            assert summary["output"] == {}
            assert summary["mcp_projection"]["output_complete"] is False
            assert summary["mcp_projection"]["omitted_output_fields"][0]["path"] == (
                "/output"
            )
            assert isinstance(summary_result.structured_content, dict)
            assert summary_result.structured_content["output"]["result"]["value"] == (
                "6"
            )

    asyncio.run(scenario())


def test_mcp_resource_links_read_exact_artifacts_and_fail_closed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client
        from mcp.shared.exceptions import MCPError

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            link = next(
                block for block in result.content if block.type == "resource_link"
            )
            resource = await client.read_resource(link.uri)
            payload = json.loads(resource.contents[0].text)
            assert payload["artifact_uri"] == link.uri
            assert payload["manifest"]["object_digest"].startswith("sha256:")

            payload_digest = payload["manifest"]["payload_digest"]
            payload_hex = payload_digest.removeprefix("sha256:")
            payload_blob = (
                tmp_path / "blobs" / "sha256" / payload_hex[:2] / payload_hex[2:]
            )
            payload_blob.write_bytes(b'{"tampered":true}')
            with pytest.raises(MCPError):
                await client.read_resource(link.uri)

            missing_uri = "artifact://sha256/" + ("f" * 64)
            with pytest.raises(MCPError):
                await client.read_resource(missing_uri)

            tampered_digest = "0" if link.uri[-1] != "0" else "1"
            tampered_uri = link.uri[:-1] + tampered_digest
            with pytest.raises(MCPError):
                await client.read_resource(tampered_uri)

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


def test_mcp_no_retrieval_policy_is_operator_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        policy = CapabilityPolicy(profile="COMPUTE_VERIFY_NO_RETRIEVAL")
        async with Client(
            create_server(tmp_path, capability_policy=policy),
            raise_exceptions=True,
        ) as client:
            resource = await client.read_resource("capability://catalog")
            catalog = json.loads(resource.contents[0].text)
            assert catalog["policy_profile"] == "COMPUTE_VERIFY_NO_RETRIEVAL"
            assert catalog["policy_digest"] == policy.digest
            assert "knowledge.search" not in {
                descriptor["capability_id"] for descriptor in catalog["capabilities"]
            }
            assert all(
                "retrieval" not in descriptor["tags"]
                for descriptor in catalog["capabilities"]
            )

            denied = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "knowledge.search",
                    "mode": "EXPLORE",
                    "payload": {"query": "counterexample", "limit": 5},
                },
            )
            result = json.loads(denied.content[0].text)
            assert result["execution"]["status"] == "ERROR"
            assert result["output"]["error"]["code"] == "CAPABILITY_POLICY_DENIED"
            assert result["assurance"]["level"] != "VERIFIED"
            assert result["diagnostics"][0]["details"] == {
                "policy_profile": "COMPUTE_VERIFY_NO_RETRIEVAL",
                "policy_digest": policy.digest,
                "reasons": ["capability_id_denied", "tag_denied"],
                "checker_authorization_affected": False,
            }

    asyncio.run(scenario())


def test_mcp_compact_capability_index_is_searchable_and_paginated(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            resource_result = await client.read_resource("capability://catalog")
            full_catalog = json.loads(resource_result.contents[0].text)
            all_ids = {
                descriptor["capability_id"]
                for descriptor in full_catalog["capabilities"]
            }

            listed = await client.call_tool(
                "capability.describe",
                {"limit": 20},
            )
            index = json.loads(listed.content[0].text)
            assert len(listed.content[0].text.encode("utf-8")) <= 16 * 1024
            assert index["catalog_digest"].startswith("sha256:")
            assert index["policy_digest"].startswith("sha256:")
            assert index["response_byte_limit"] == 16 * 1024
            assert len(index["matches"]) <= 20
            indexed_ids = {
                descriptor["capability_id"] for descriptor in index["matches"]
            }
            assert all(
                "input_schema" not in descriptor for descriptor in index["matches"]
            )
            cursor = index["next_cursor"]
            while cursor is not None:
                next_page = await client.call_tool(
                    "capability.describe",
                    {"cursor": cursor, "limit": 20},
                )
                page = json.loads(next_page.content[0].text)
                assert len(next_page.content[0].text.encode("utf-8")) <= 16 * 1024
                assert page["catalog_digest"] == index["catalog_digest"]
                indexed_ids.update(
                    descriptor["capability_id"] for descriptor in page["matches"]
                )
                cursor = page["next_cursor"]
            assert indexed_ids == all_ids

            searched = await client.call_tool(
                "capability.describe",
                {
                    "query": "SAT UNSAT proof",
                    "input_kind": "STRUCTURED_REQUEST",
                },
            )
            search_index = json.loads(searched.content[0].text)
            search_ids = {
                descriptor["capability_id"] for descriptor in search_index["matches"]
            }
            expected_sat_ids = {
                "sat.cnf.materialize",
                "sat.unsat_proof.find",
                "sat.unsat_proof.verify",
            }.intersection(all_ids)
            assert expected_sat_ids.issubset(search_ids)

            coloring_search = await client.call_tool(
                "capability.describe",
                {
                    "query": (
                        "finite coloring forbidden monochromatic triples exact "
                        "finite existence certified exhaustive search"
                    ),
                    "limit": 20,
                },
            )
            coloring_ids = {
                descriptor["capability_id"]
                for descriptor in json.loads(coloring_search.content[0].text)["matches"]
            }
            assert expected_sat_ids.issubset(coloring_ids)

            materialize_description = await client.call_tool(
                "capability.describe",
                {
                    "capability_id": "sat.cnf.materialize",
                    "view": "CONTRACT",
                },
            )
            materialize = json.loads(materialize_description.content[0].text)
            assert materialize["invocations"][0]["name"] == "finite-coloring-cnf"
            assert (
                materialize["synchronous_execution"]["remote_safe_wall_seconds_max"]
                == 150
            )
            assert {
                item["capability_id"] for item in materialize["related_capabilities"]
            }.issuperset(expected_sat_ids - {"sat.cnf.materialize"})

            first_page = await client.call_tool(
                "capability.describe",
                {"limit": 20},
            )
            first = json.loads(first_page.content[0].text)
            assert len(first["matches"]) <= 20
            assert first["next_cursor"] is not None
            second_page = await client.call_tool(
                "capability.describe",
                {"cursor": first["next_cursor"], "limit": 20},
            )
            second = json.loads(second_page.content[0].text)
            assert {
                descriptor["capability_id"] for descriptor in first["matches"]
            }.isdisjoint(
                descriptor["capability_id"] for descriptor in second["matches"]
            )

            invalid_cursor = await client.call_tool(
                "capability.describe",
                {
                    "query": "definitely-no-matching-capability",
                    "cursor": first["next_cursor"],
                    "limit": 20,
                },
            )
            invalid = json.loads(invalid_cursor.content[0].text)
            assert invalid["error"]["code"] == "INVALID_CURSOR"

    asyncio.run(scenario())
