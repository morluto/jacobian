from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from mcp.shared.exceptions import MCPError

from jacobian.adapters.mcp.guidance import OPERATING_GUIDE
from jacobian.adapters.mcp.server import create_server
from jacobian.contracts.capabilities import CapabilityDescriptor

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


def test_math_tool_surface_is_consistent_across_discovery(tmp_path: Path) -> None:
    server = create_server(tmp_path)
    assert server.instructions is not None
    assert "math.find" in server.instructions

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == {"math.find", "math.run"}
            assert tools["math.find"].title == "Find an exact mathematical operation"
            assert tools["math.run"].title == "Run a mathematical operation"
            assert "math.find" in (tools["math.run"].description or "")

            discovery = await client.get_prompt(
                "jacobian-discover",
                {"task": "Find a finite counterexample."},
            )
            discovery_text = discovery.messages[0].content.text
            assert "math.find" in discovery_text
            assert "math.run" in discovery_text

            evidence = await client.get_prompt(
                "jacobian-check-evidence",
                {"claim": "The candidate satisfies the defining identity."},
            )
            evidence_text = evidence.messages[0].content.text
            assert "math.find" in evidence_text

            described = await client.call_tool(
                "math.find",
                {"capability_id": "integer.compute.gcd", "view": "CONTRACT"},
            )
            assert described.structured_content is not None
            invocations = described.structured_content["invocations"]
            assert invocations
            assert {item["tool"] for item in invocations} == {"math.run"}

            absent = await client.call_tool(
                "math.find",
                {"query": "quuxonium frobnicator"},
            )
            assert absent.structured_content is not None
            recovery_tools = {
                item["tool"]
                for item in absent.structured_content["available_recovery_paths"]
                if "tool" in item
            }
            assert recovery_tools == {"math.find"}

    asyncio.run(scenario())


def test_math_find_exposes_bounded_examples_and_actionable_contract_text(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            discovered = await client.call_tool(
                "math.find",
                {
                    "query": "compute an exact matrix determinant",
                    "domain": "matrix",
                    "mode": "EXPLORE",
                    "limit": 3,
                },
            )
            assert isinstance(discovered.structured_content, dict)
            compute = next(
                match
                for match in discovered.structured_content["matches"]
                if match["capability_id"] == "matrix.determinant.compute"
            )
            example = compute["invocation_example"]
            assert len(json.dumps(example).encode("utf-8")) <= 2 * 1024
            discovery_text = json.loads(discovered.content[0].text)
            compute_text = next(
                match
                for match in discovery_text["matches"]
                if match["capability_id"] == "matrix.determinant.compute"
            )
            for decision_field in (
                "description",
                "accepted_input_kinds",
                "output_schema_summary",
                "scope",
                "assurance_ceiling",
                "related_capabilities",
                "invocation_example",
            ):
                assert compute_text[decision_field] == compute[decision_field]

            verifier_discovery = await client.call_tool(
                "math.find",
                {
                    "query": "independently verify an exact determinant artifact",
                    "domain": "matrix",
                    "mode": "VERIFY",
                    "limit": 3,
                },
            )
            assert isinstance(verifier_discovery.structured_content, dict)
            verifier = next(
                match
                for match in verifier_discovery.structured_content["matches"]
                if match["capability_id"] == "matrix.determinant.verify"
            )
            assert "invocation_example" not in verifier
            assert verifier["input_schema_summary"] == {
                "type": "object",
                "required": ["input", "candidate"],
                "property_names": ["candidate", "input"],
            }
            verifier_text = next(
                match
                for match in json.loads(verifier_discovery.content[0].text)["matches"]
                if match["capability_id"] == "matrix.determinant.verify"
            )
            assert (
                verifier_text["input_schema_summary"]
                == verifier["input_schema_summary"]
            )

            compute_contract = await client.call_tool(
                "math.find",
                {
                    "capability_id": "matrix.determinant.compute",
                    "view": "CONTRACT",
                },
            )
            assert isinstance(compute_contract.structured_content, dict)
            schema = compute_contract.structured_content["capability"]["input_schema"]
            assert Draft202012Validator(schema).is_valid(example["payload"])

            verify_contract = await client.call_tool(
                "math.find",
                {
                    "capability_id": "matrix.determinant.verify",
                    "view": "CONTRACT",
                },
            )
            verify_text = json.loads(verify_contract.content[0].text)
            assert verify_text["capability"]["input_schema"]["required"] == [
                "input",
                "candidate",
            ]

    asyncio.run(scenario())


def test_math_find_compacts_related_capabilities_deterministically(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.adapters.mcp import projections
    from jacobian.adapters.mcp.tools import _find_result

    target_id = "polynomial.integer.compute.gcd"
    catalog_ids = tuple(
        descriptor.capability_id
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id != target_id
    )
    monkeypatch.setitem(
        projections._RELATED_CAPABILITIES,
        target_id,
        tuple(
            (capability_id, f"compatible exact outcome {index:04d} " + "x" * 80)
            for index, capability_id in enumerate(catalog_ids)
        ),
    )
    byte_limit = 8 * 1024
    monkeypatch.setattr(
        projections,
        "CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT",
        byte_limit,
    )

    def discover() -> dict[str, Any]:
        return projections._capability_discovery_response(
            fresh_complete_runtime,
            query=target_id,
            domain=None,
            mode=None,
            input_kind=None,
            artifact_type=None,
            limit=1,
            cursor=None,
        )

    first = discover()
    second = discover()

    assert first == second
    assert first["matches"][0]["capability_id"] == target_id
    assert len(projections._mcp_text_json_bytes(first)) <= byte_limit
    assert first["related_capabilities_truncated"] is True
    assert first["truncation_reason"] == "BYTE_LIMIT"
    related = first["matches"][0]["related_capabilities"]
    assert [item["capability_id"] for item in related] == sorted(catalog_ids)[
        : len(related)
    ]

    tool_result = _find_result(first)
    text_result = json.loads(tool_result.content[0].text)
    assert tool_result.structured_content is not None
    assert tool_result.structured_content["related_capabilities_truncated"] is True
    assert text_result["related_capabilities_truncated"] is True
    assert text_result["truncation_reason"] == "BYTE_LIMIT"


def test_math_find_compacts_relationships_before_ranked_discovery_data(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.adapters.mcp import projections

    arguments = {
        "query": "polynomial",
        "domain": None,
        "mode": None,
        "input_kind": None,
        "artifact_type": None,
        "limit": 5,
        "cursor": None,
    }
    baseline = projections._capability_discovery_response(
        fresh_complete_runtime, **arguments
    )
    baseline_match_ids = [item["capability_id"] for item in baseline["matches"]]
    baseline_domains = baseline["available_domains"]
    catalog_ids = tuple(
        descriptor.capability_id
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
    )
    for target_id in baseline_match_ids:
        monkeypatch.setitem(
            projections._RELATED_CAPABILITIES,
            target_id,
            tuple(
                (capability_id, "compatible exact outcome " + "x" * 80)
                for capability_id in catalog_ids
                if capability_id != target_id
            ),
        )
    monkeypatch.setattr(
        projections,
        "CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT",
        len(projections._mcp_text_json_bytes(baseline)) + 512,
    )

    compacted = projections._capability_discovery_response(
        fresh_complete_runtime, **arguments
    )

    assert [
        item["capability_id"] for item in compacted["matches"]
    ] == baseline_match_ids
    assert compacted["available_domains"] == baseline_domains
    assert compacted["related_capabilities_truncated"] is True
    assert compacted["match_metadata_truncated"] is False


def test_math_find_accounts_for_fixed_metadata_before_compacting_relationships(
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.adapters.mcp import projections

    target_id = "polynomial.integer.compute.gcd"
    arguments = {
        "query": target_id,
        "domain": None,
        "mode": None,
        "input_kind": None,
        "artifact_type": None,
        "limit": 1,
        "cursor": None,
    }
    related_id = next(
        descriptor.capability_id
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id != target_id
    )
    monkeypatch.setitem(
        projections._RELATED_CAPABILITIES,
        target_id,
        ((related_id, "compatible exact outcome"),),
    )
    monkeypatch.setattr(
        projections,
        "CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT",
        10_000,
    )
    candidate = projections._capability_discovery_response(
        fresh_complete_runtime, **arguments
    )
    candidate_domains = candidate["available_domains"]
    monkeypatch.setattr(
        projections,
        "CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT",
        len(projections._mcp_text_json_bytes(candidate)) - 2,
    )

    compacted = projections._capability_discovery_response(
        fresh_complete_runtime, **arguments
    )

    assert compacted["matches"][0]["related_capabilities"] == []
    assert compacted["related_capabilities_truncated"] is True
    assert compacted["available_domains"] == candidate_domains
    assert compacted["available_domains_truncated"] is False


@pytest.mark.parametrize("view", ["CONTRACT", "FULL"])
def test_math_find_compacts_exact_inspection_relationships(
    tmp_path: Path,
    fresh_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
    view: str,
) -> None:
    from jacobian.adapters.mcp import projections

    target_id = "polynomial.integer.compute.gcd"
    monkeypatch.setitem(
        projections._RELATED_CAPABILITIES,
        target_id,
        tuple(
            (descriptor.capability_id, "compatible exact outcome " + "x" * 200)
            for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
            if descriptor.capability_id != target_id
        ),
    )

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.find", {"capability_id": target_id, "view": view}
            )

        assert result.structured_content is not None
        structured = result.structured_content
        text_result = json.loads(result.content[0].text)
        assert (
            len(
                projections._mcp_text_json_bytes(
                    structured.get("related_capabilities", [])
                )
            )
            <= structured["related_capabilities_byte_limit"]
        )
        assert structured["related_capabilities_truncated"] is True
        assert structured["truncation_reason"] == "BYTE_LIMIT"
        assert "response_byte_limit" not in structured
        assert "related_capabilities" not in structured["capability"]
        assert text_result["related_capabilities_truncated"] is True
        assert text_result["truncation_reason"] == "BYTE_LIMIT"
        assert (
            text_result["related_capabilities_byte_limit"]
            == structured["related_capabilities_byte_limit"]
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("view", ["CONTRACT", "FULL"])
def test_math_find_does_not_advertise_a_whole_response_limit_for_large_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    view: str,
) -> None:
    from jacobian.adapters.mcp import projections, tools

    original_view = projections._capability_descriptor_view

    def oversized_view(*args, **kwargs):
        projected = original_view(*args, **kwargs)
        if kwargs["view"] == view:
            projected["input_schema"] = {
                "type": "object",
                "x-contract-padding": "x"
                * projections.CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT,
            }
        return projected

    monkeypatch.setattr(tools, "_capability_descriptor_view", oversized_view)

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.find",
                {
                    "capability_id": "polynomial.integer.compute.gcd",
                    "view": view,
                },
            )

        assert result.structured_content is not None
        structured = result.structured_content
        assert structured["view"] == view
        assert structured["capability"]["input_schema"]["x-contract-padding"]
        assert "response_byte_limit" not in structured
        assert (
            len(
                projections._mcp_text_json_bytes(
                    structured.get("related_capabilities", [])
                )
            )
            <= structured["related_capabilities_byte_limit"]
        )

    asyncio.run(scenario())


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
                "ranking is deterministic retrieval"
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
                "mode",
                "input_kind",
                "artifact_type",
                "limit",
                "cursor",
                "view",
            }
            assert set(tools["math.run"].input_schema["properties"]) == {
                "capability_id",
                "payload",
                "mode",
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
            assert operation_card["assurance_ceiling"] in {"COMPUTED", "VERIFIED"}
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


def test_mcp_describes_and_invokes_capabilities(tmp_path: Path) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            described = await client.call_tool(
                "math.find",
                {"capability_id": "integer.compute.gcd", "view": "CONTRACT"},
            )
            assert isinstance(described.structured_content, dict)
            contract = described.structured_content
            assert contract["view"] == "CONTRACT"
            assert contract["capability"]["capability_id"] == "integer.compute.gcd"
            assert contract["capability"]["provider_runtime"]["digest"].startswith(
                "sha256:"
            )
            assert "configuration" not in contract["capability"]["provider_runtime"]
            assert "output_schema" not in contract["capability"]
            assert "output_schema_summary" in contract["capability"]

            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content["artifact_uris"] == []
            response = json.loads(result.content[0].text)
            assert response["execution"]["status"] == "COMPLETED"
            assert response["assurance"]["level"] == "COMPUTED"
            assert isinstance(result.structured_content, dict)
            assert "mcp_projection" not in result.structured_content
            assert result.structured_content["output"] == response["output"]
            for semantic_field in (
                "scope",
                "completeness",
                "relationships",
                "obligations",
                "assurance",
            ):
                assert (
                    response[semantic_field]
                    == result.structured_content[semantic_field]
                )
            runtime = contract["capability"]["provider_runtime"]
            assert (
                result.structured_content["provider"]
                == contract["capability"]["provider"]
            )
            assert result.structured_content["provider_digest"] == runtime["digest"]

            matching_description = await client.call_tool(
                "math.find",
                {
                    "capability_id": ("graph.invariant.maximum_matching.compute"),
                    "view": "CONTRACT",
                },
            )
            assert isinstance(matching_description.structured_content, dict)
            matching_contract = matching_description.structured_content
            assert matching_contract["capability"]["version"] == "3"
            assert matching_contract["invocations"][0]["name"] == ("triangle_with_tail")
            assert matching_contract["related_capabilities"] == [
                {
                    "capability_id": ("graph.invariant.maximum_matching.verify"),
                    "kind": "INDEPENDENT_VERIFIER",
                    "relationship": "independently verify this exact producer result",
                }
            ]

            reliability_verifier_discovery = await client.call_tool(
                "math.find",
                {
                    "query": (
                        "independently verify exact graph reliability terminal "
                        "connection probability edge subset enumeration"
                    ),
                    "domain": "graph",
                    "mode": "VERIFY",
                    "limit": 10,
                },
            )
            assert isinstance(reliability_verifier_discovery.structured_content, dict)
            assert "probability.graph_reliability.connection_probability.verify" in {
                match["capability_id"]
                for match in reliability_verifier_discovery.structured_content[
                    "matches"
                ]
            }

            modular_compute = await client.call_tool(
                "math.find",
                {
                    "capability_id": "modular.polynomial_residue_image.compute",
                    "view": "CONTRACT",
                },
            )
            modular_verify = await client.call_tool(
                "math.find",
                {
                    "capability_id": "modular.polynomial_residue_image.verify",
                    "view": "CONTRACT",
                },
            )
            assert isinstance(modular_compute.structured_content, dict)
            assert isinstance(modular_verify.structured_content, dict)
            assert {
                item["capability_id"]
                for item in modular_compute.structured_content["related_capabilities"]
            } == {"modular.polynomial_residue_image.verify"}
            assert {
                item["capability_id"]
                for item in modular_verify.structured_content["related_capabilities"]
            } == {"modular.polynomial_residue_image.compute"}

            unknown = await client.call_tool(
                "math.run",
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
            assert "available_capability_ids" not in unknown_result["output"]
            assert len(unknown.content[0].text.encode("utf-8")) < 2_048
            assert isinstance(unknown.structured_content, dict)
            output = unknown.structured_content["output"]
            assert "available_capability_ids" not in output
            assert len(output["nearby_capability_ids"]) <= 5
            assert output["available_recovery_paths"][-1] == {
                "action": "inspect_catalog",
                "resource_uri": "capability://catalog",
            }
            assert unknown_result["assurance"]["level"] != "VERIFIED"

    asyncio.run(scenario())


def test_mcp_inline_results_do_not_emit_resource_links(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "integer.compute.gcd",
                    "mode": "EXPLORE",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content["artifact_uris"] == []
            assert [
                block for block in result.content if block.type == "resource_link"
            ] == []

    asyncio.run(scenario())


def test_mcp_materialized_results_emit_readable_native_resource_links(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "sat.cnf.materialize",
                    "mode": "EXPLORE",
                    "payload": {
                        "variable_names": ["x"],
                        "clauses": [[1]],
                    },
                },
            )
            assert isinstance(result.structured_content, dict)
            artifact_uris = result.structured_content["artifact_uris"]
            links = [block for block in result.content if block.type == "resource_link"]
            assert [str(link.uri) for link in links] == artifact_uris
            assert [link.name for link in links] == artifact_uris
            assert all(link.mime_type == "application/json" for link in links)

            resource = await client.read_resource(links[0].uri)
            envelope = json.loads(resource.contents[0].text)
            assert envelope["artifact_uri"] == artifact_uris[0]
            assert envelope["payload"]["clauses"] == [{"literals": [1]}]

    asyncio.run(scenario())


def test_mcp_exact_description_layers_summary_contract_and_full_views(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            summary_result = await client.call_tool(
                "math.find",
                {"capability_id": "polynomial.expression.normalize"},
            )
            contract_result = await client.call_tool(
                "math.find",
                {
                    "capability_id": "polynomial.expression.normalize",
                    "view": "CONTRACT",
                },
            )
            full_result = await client.call_tool(
                "math.find",
                {
                    "capability_id": "polynomial.expression.normalize",
                    "view": "FULL",
                },
            )
            assert isinstance(summary_result.structured_content, dict)
            assert isinstance(contract_result.structured_content, dict)
            assert isinstance(full_result.structured_content, dict)
            summary = summary_result.structured_content
            contract = contract_result.structured_content
            full = full_result.structured_content

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
            summary_text = json.loads(summary_result.content[0].text)
            contract_text = json.loads(contract_result.content[0].text)
            full_text = json.loads(full_result.content[0].text)
            assert "input_schema" not in summary_text["capability"]
            assert (
                contract_text["capability"]["input_schema"]
                == contract["capability"]["input_schema"]
            )
            assert "output_schema" not in full_text["capability"]
            assert len(full_result.content[0].text) < len(
                json.dumps(full, separators=(",", ":"))
            )

    asyncio.run(scenario())
