"""Executable contract for the stable MCP tool surface."""

from __future__ import annotations

from jacobian.adapters.mcp.server import JacobianCoreExtension


def test_core_extension_exposes_exactly_the_stable_capability_tools() -> None:
    extension = JacobianCoreExtension(None, None)
    assert extension.identifier == "io.jacobian/core"
    assert extension.settings() == {"version": "1"}
    assert tuple(binding.kwargs["name"] for binding in extension.tools()) == (
        "capability.describe",
        "capability.invoke",
    )
