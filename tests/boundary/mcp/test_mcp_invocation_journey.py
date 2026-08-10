"""Owned MCP smoke journey: live SDK find → run without complete-runtime fixtures.

``create_server(tmp_path)`` installs a fresh server-owned runtime. Keep this
module small; do not grow ordinary projection or capability matrices here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from jacobian.adapters.mcp.server import create_server

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


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
                if semantic_field in response:
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
