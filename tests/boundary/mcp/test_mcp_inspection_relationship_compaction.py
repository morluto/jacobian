"""Live MCP inspection compaction against the installed built-in catalog."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jacobian.adapters.mcp.server import create_server
from jacobian.domains.builtins import build_builtin_domain_bundles


def test_math_find_compacts_exact_inspection_relationships(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.adapters.mcp import projections

    target_id = "polynomial.integer.compute.gcd"
    catalog_ids = tuple(
        operation.capability_id
        for bundle in build_builtin_domain_bundles()
        for operation in bundle.capabilities
        if operation.capability_id != target_id
    )
    monkeypatch.setitem(
        projections._RELATED_CAPABILITIES,
        target_id,
        tuple(
            (capability_id, "compatible exact outcome " + "x" * 200)
            for capability_id in catalog_ids
        ),
    )

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            for view in ("CONTRACT", "FULL"):
                result = await client.call_tool(
                    "math.find", {"capability_id": target_id, "view": view}
                )

                assert result.structured_content is not None, view
                structured = result.structured_content
                text_result = json.loads(result.content[0].text)
                assert (
                    len(
                        projections._mcp_text_json_bytes(
                            structured.get("related_capabilities", [])
                        )
                    )
                    <= structured["related_capabilities_byte_limit"]
                ), view
                assert structured["related_capabilities_truncated"] is True, view
                assert structured["truncation_reason"] == "BYTE_LIMIT", view
                assert "response_byte_limit" not in structured, view
                assert "related_capabilities" not in structured["capability"], view
                assert text_result["related_capabilities_truncated"] is True, view
                assert text_result["truncation_reason"] == "BYTE_LIMIT", view
                assert (
                    text_result["related_capabilities_byte_limit"]
                    == structured["related_capabilities_byte_limit"]
                ), view

    asyncio.run(scenario())
