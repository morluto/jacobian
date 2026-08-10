from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from jacobian.adapters.mcp.server import create_server

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


def test_math_find_compacts_related_capabilities_deterministically(
    attached_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.adapters.mcp import projections
    from jacobian.adapters.mcp.tools import _find_result

    target_id = "polynomial.integer.compute.gcd"
    catalog_ids = tuple(
        descriptor.capability_id
        for descriptor in attached_complete_runtime.core.capabilities.catalog().capabilities
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
            attached_complete_runtime,
            query=target_id,
            domain=None,
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
    attached_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.adapters.mcp import projections

    arguments = {
        "query": "polynomial",
        "domain": None,
        "input_kind": None,
        "artifact_type": None,
        "limit": 5,
        "cursor": None,
    }
    baseline = projections._capability_discovery_response(
        attached_complete_runtime, **arguments
    )
    baseline_match_ids = [item["capability_id"] for item in baseline["matches"]]
    baseline_domains = baseline["available_domains"]
    catalog_ids = tuple(
        descriptor.capability_id
        for descriptor in attached_complete_runtime.core.capabilities.catalog().capabilities
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
        attached_complete_runtime, **arguments
    )

    assert [
        item["capability_id"] for item in compacted["matches"]
    ] == baseline_match_ids
    assert compacted["available_domains"] == baseline_domains
    assert compacted["related_capabilities_truncated"] is True
    assert compacted["match_metadata_truncated"] is False


def test_math_find_accounts_for_fixed_metadata_before_compacting_relationships(
    attached_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.adapters.mcp import projections

    target_id = "polynomial.integer.compute.gcd"
    arguments = {
        "query": target_id,
        "domain": None,
        "input_kind": None,
        "artifact_type": None,
        "limit": 1,
        "cursor": None,
    }
    related_id = next(
        descriptor.capability_id
        for descriptor in attached_complete_runtime.core.capabilities.catalog().capabilities
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
        attached_complete_runtime, **arguments
    )
    candidate_domains = candidate["available_domains"]
    monkeypatch.setattr(
        projections,
        "CAPABILITY_DISCOVERY_RESPONSE_BYTE_LIMIT",
        len(projections._mcp_text_json_bytes(candidate)) - 2,
    )

    compacted = projections._capability_discovery_response(
        attached_complete_runtime, **arguments
    )

    assert compacted["matches"][0]["related_capabilities"] == []
    assert compacted["related_capabilities_truncated"] is True
    assert compacted["available_domains"] == candidate_domains
    assert compacted["available_domains_truncated"] is False


@pytest.mark.parametrize("view", ["CONTRACT", "FULL"])
def test_math_find_compacts_exact_inspection_relationships(
    tmp_path: Path,
    attached_complete_runtime,
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
            for descriptor in attached_complete_runtime.core.capabilities.catalog().capabilities
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
