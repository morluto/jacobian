"""Availability and missing-runtime recovery through the real MCP adapter."""

from __future__ import annotations

import asyncio
import copy
import json
import shutil

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "operation_id",
    [
        "polynomial.ideal.radical.compute",
        "algebraic_geometry.projective_plane_curve.singularity_profile.compute",
    ],
)
def test_missing_singular_is_inspectable_and_actionable(
    monkeypatch: pytest.MonkeyPatch,
    direct: bool,
    operation_id: str,
) -> None:
    original = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name == "Singular" else original(name)
    )
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    payload = copy.deepcopy(operation.examples[0].input)
    if operation_id.startswith("algebraic_geometry."):
        for term in payload["polynomial"]["polynomial"]["terms"]:
            term["exponents"] = [2 * exponent for exponent in term["exponents"]]
    catalog = Catalog((operation,))
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )

    async def scenario() -> None:
        async with Client(server, raise_exceptions=False) as client:
            inspection = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": operation_id,
                    }
                },
            )
            assert inspection.structured_content is not None
            assert inspection.structured_content["operation"][
                "runtime_requirements"
            ] == ["singular"]
            availability = inspection.structured_content["backend_availability"]
            assert availability[0]["status"] == "MISSING"
            result = await client.call_tool(
                operation_id if direct else "math.run",
                payload
                if direct
                else {
                    "operation_id": operation_id,
                    "payload": payload,
                },
            )
            assert result.is_error
            messages = [item.text for item in result.content if item.type == "text"]
            tool_name = operation_id if direct else "math.run"
            diagnostic = json.loads(
                messages[0].removeprefix(f"Error executing tool {tool_name}: ")
            )
            assert diagnostic["code"] == "BACKEND_UNAVAILABLE"
            assert diagnostic["backend"] == "singular"
            assert diagnostic["required_version"] == "4.4.x"
            assert "server operator" in diagnostic["hint"]

    asyncio.run(scenario())
