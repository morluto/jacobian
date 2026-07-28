from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from jacobian.adapters.mcp.guidance import OPERATING_GUIDE
from jacobian.adapters.mcp.server import (
    WORKSPACE_TOOL_NAMES,
    _public_tool_error,
    _request_trace_digest,
    create_server,
)
from jacobian.capabilities import CapabilityPolicy
from jacobian.contracts.capabilities import CapabilityDescriptor

CAPABILITY_TOOL_NAMES = {"capability.describe", "capability.invoke"}
MCP_TOOL_NAMES = CAPABILITY_TOOL_NAMES | WORKSPACE_TOOL_NAMES


def test_mcp_trace_correlation_hashes_headers_without_retaining_them() -> None:
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

    class RequestContext:
        def __init__(self) -> None:
            self.headers = {"traceparent": traceparent}
            self.request_id = "private-request-id"

    digest, source = _request_trace_digest(RequestContext())

    assert digest == hashlib.sha256(traceparent.encode()).hexdigest()[:8]
    assert source == "traceparent"
    assert traceparent not in digest
    assert "private-request-id" not in digest


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
            assert set(describe_schema["properties"]) == {
                "capability_id",
                "query",
                "domain",
                "mode",
                "limit",
                "cursor",
                "view",
            }
            assert describe_schema["additionalProperties"] is False
            assert (
                tools["capability.invoke"].input_schema["additionalProperties"] is False
            )
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
            resource_uris = {str(resource.uri) for resource in resources.resources}
            assert "jacobian://instructions" in resource_uris
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


def test_mcp_workspace_schema_aliases_and_fail_closed_round_trip(
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
            assert write_schema["additionalProperties"] is False
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
            assert "OPEN_GOAL" in finding_kinds
            assert "PROBLEM" not in finding_kinds
            assert "SUCCEEDED" in attempt_outcomes
            mark_schema = write_schema["$defs"]["WorkspaceMarkDraft"]
            assert "summary" in mark_schema["properties"]
            assert mark_schema["oneOf"]

            alias_payload = {
                "workspace_id": "workspace://" + ("0" * 32),
                "branch_id": "branch://" + ("0" * 32),
                "base_revision": "revision://" + ("0" * 32),
                "idempotency_key": "schema-aliases-001",
                "findings": [
                    {
                        "client_ref": "G1",
                        "kind": "OPEN_GOAL",
                        "title": "Open work",
                        "body": "A documented finding-kind alias.",
                    }
                ],
                "attempts": [
                    {
                        "client_ref": "T1",
                        "target_ref": "G1",
                        "method": "direct",
                        "outcome": "SUCCEEDED",
                        "summary": "A documented attempt-outcome alias.",
                    }
                ],
                "marks": [
                    {
                        "client_ref": "M1",
                        "target_ref": "G1",
                        "state": "CLOSED",
                        "summary": "A documented mark-reason alias.",
                    }
                ],
            }
            write_validator = Draft202012Validator(write_schema)
            assert write_validator.is_valid(alias_payload), list(
                write_validator.iter_errors(alias_payload)
            )
            assert not write_validator.is_valid(
                {
                    **alias_payload,
                    "marks": [
                        {
                            "client_ref": "M1",
                            "target_ref": "G1",
                            "state": "CLOSED",
                            "reason": "Canonical reason.",
                            "summary": "Conflicting alias.",
                        }
                    ],
                }
            )
            assert not write_validator.is_valid(
                {
                    **alias_payload,
                    "findings": [
                        {
                            "client_ref": "P2",
                            "kind": "PROBLEM",
                            "title": "Hidden second problem",
                            "body": "Only workspace.open creates the problem.",
                        }
                    ],
                    "attempts": [],
                    "marks": [],
                }
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
            rejected = json.loads(rejected_result.content[0].text)
            assert rejected["error"]["code"] == "INVALID_INPUT"

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
            second_problem = json.loads(second_problem_result.content[0].text)
            assert second_problem["error"]["code"] == "INVALID_INPUT"

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
                            "kind": "OPEN_GOAL",
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
                            "outcome": "SUCCEEDED",
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

            conflicting_mark_result = await client.call_tool(
                "workspace.write",
                {
                    "idempotency_key": "mcp-workspace-mark-conflict-001",
                    "workspace_id": opened["workspace_id"],
                    "branch_id": opened["branch_id"],
                    "base_revision": written["revision_id"],
                    "marks": [
                        {
                            "client_ref": "M0",
                            "target_ref": written["id_map"]["A1"],
                            "state": "RETRACTED",
                            "reason": "Canonical reason.",
                            "summary": "Conflicting alias.",
                        }
                    ],
                },
            )
            assert conflicting_mark_result.is_error is True
            conflicting_mark = json.loads(conflicting_mark_result.content[0].text)
            assert conflicting_mark["error"]["code"] == "INVALID_INPUT"

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
                            "summary": "The temporary scope was withdrawn.",
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
            response = json.loads(result.content[0].text)
            assert response["execution"]["status"] == "COMPLETED"
            assert response["assurance"]["level"] == "COMPUTED"
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
            compact_result = await client.call_tool(
                "capability.describe",
                {
                    "capability_id": "polynomial.expression.normalize",
                    "view": "COMPACT",
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
            compact = json.loads(compact_result.content[0].text)
            full = json.loads(full_result.content[0].text)

            assert summary["view"] == "SUMMARY"
            assert "input_schema" not in summary["capability"]
            assert summary["capability"]["input_schema_summary"]["type"] == "object"
            assert summary["capability"]["has_invocation_examples"] is True
            assert "invocations" not in summary
            assert "CONTRACT" in summary["next_views"]
            assert "all-orders" in summary["scope_rule"]["bounded_repetition"]
            assert contract["view"] == "CONTRACT"
            assert contract["capability"]["input_schema"]["type"] == "object"
            assert contract["invocations"]
            assert compact["view"] == "COMPACT"
            assert compact["capability"] == contract["capability"]
            assert compact["invocations"] == contract["invocations"]
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
                {"query": "SAT UNSAT proof"},
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


def test_mcp_logs_bounded_tool_metrics_without_arguments(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jacobian.adapters.mcp.server")

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            await client.call_tool(
                "capability.describe",
                {"query": "private-query-marker"},
            )
            failed = await client.call_tool(
                "capability.invoke",
                {
                    "capability_id": "missing.capability",
                    "mode": "EXPLORE",
                    "payload": {"private": "private-payload-marker"},
                },
            )
            response = json.loads(failed.content[0].text)
            assert response["execution"]["status"] == "ERROR"

    asyncio.run(scenario())

    messages = [record.getMessage() for record in caplog.records]
    metric = next(
        message
        for message in messages
        if "MCP tool call tool=capability.describe status=success" in message
    )
    assert "duration_ms=" in metric
    assert "response_bytes=" in metric
    assert "argument_digest=sha256:" in metric
    assert "private-query-marker" not in metric
    attempt = next(
        message
        for message in messages
        if "MCP capability attempt" in message
        and "capability_id=missing.capability" in message
    )
    assert "execution_status=ERROR" in attempt
    assert "diagnostic_codes=UNKNOWN_CAPABILITY" in attempt
    assert "trace_digest=" in attempt
    assert "argument_digest=sha256:" in attempt
    assert "private-payload-marker" not in attempt


def test_mcp_tool_failures_return_safe_actionable_errors(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=False) as client:
            unknown_capability = await client.call_tool(
                "capability.describe", {"capability_id": "missing.capability"}
            )
            response = json.loads(unknown_capability.content[0].text)
            assert response["error"]["code"] == "UNKNOWN_CAPABILITY"
            assert "search installed capabilities" in response["error"]["hint"]

    asyncio.run(scenario())

    internal = json.loads(_public_tool_error("fixture", KeyError("internal")))
    assert internal["error"]["code"] == "OPERATION_FAILED"


def test_mcp_protocol_and_authentication_errors_remain_distinct(tmp_path: Path) -> None:
    from mcp.shared.exceptions import MCPError

    server = create_server(tmp_path)

    @server.tool(name="fixture.protocol-error")
    async def protocol_error() -> None:
        raise MCPError(123, "protocol action required")

    with pytest.raises(MCPError, match="protocol action required"):
        asyncio.run(server.call_tool("fixture.protocol-error", {}))

    async def scenario() -> None:
        from mcp import Client

        async with Client(
            create_server(
                tmp_path,
                tenant_isolation=True,
                allow_anonymous=False,
            ),
            raise_exceptions=False,
        ) as client:
            response = await client.call_tool("capability.describe", {})
            assert response.is_error is True
            assert '"code": "AUTHENTICATION_REQUIRED"' in response.content[0].text

    asyncio.run(scenario())


@pytest.mark.subprocess
def test_mcp_stdio_entrypoint_exposes_capability_and_workspace_tools(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client, StdioServerParameters, stdio_client

        environment = dict(os.environ)
        environment["JACOBIAN_STATE_DIR"] = str(tmp_path)
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "jacobian.adapters.mcp.server"],
            env=environment,
            cwd=Path.cwd(),
        )
        async with Client(
            stdio_client(parameters),
            raise_exceptions=True,
        ) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == MCP_TOOL_NAMES

    asyncio.run(scenario())


@pytest.mark.subprocess
def test_mcp_entrypoint_has_nonstarting_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "jacobian.adapters.mcp.server", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0
    assert "Run the Jacobian MCP server" in completed.stdout
    assert "--tool-profile" not in completed.stdout
