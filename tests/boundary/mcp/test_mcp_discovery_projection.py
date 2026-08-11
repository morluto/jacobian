from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from jacobian.adapters.mcp.server import create_server
from jacobian.contracts.capabilities import CapabilityDescriptor

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


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
                "related_capabilities",
                "invocation_example",
            ):
                assert compute_text[decision_field] == compute[decision_field]

            verifier_discovery = await client.call_tool(
                "math.find",
                {
                    "query": "independently verify an exact determinant artifact",
                    "domain": "matrix",
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
