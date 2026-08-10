from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jacobian.adapters.mcp.server import create_server


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
