"""Executable contract for the stable MCP tool surface."""

from __future__ import annotations

from jacobian.adapters.mcp.context import AppState
from jacobian.adapters.mcp.guidance import (
    MATH_FIND_DESCRIPTION,
    MATH_RUN_DESCRIPTION,
    SERVER_INSTRUCTIONS,
)
from jacobian.adapters.mcp.server import JacobianCoreExtension
from jacobian.adapters.mcp.tooling import MCPBlockingWorkerRegistry


def test_core_extension_exposes_exactly_the_stable_math_tools() -> None:
    extension = JacobianCoreExtension(
        AppState(lambda: None, MCPBlockingWorkerRegistry())  # type: ignore[arg-type]
    )
    assert extension.identifier == "io.jacobian/core"
    assert extension.settings() == {"version": "2"}
    assert tuple(binding.kwargs["name"] for binding in extension.tools()) == (
        "math.find",
        "math.run",
    )


def test_model_visible_guidance_exposes_affordances_without_research_order() -> None:
    combined = "\n".join(
        (
            SERVER_INSTRUCTIONS,
            MATH_FIND_DESCRIPTION,
            MATH_RUN_DESCRIPTION,
        )
    ).lower()
    assert "desired local mathematical outcome" in combined
    assert "not recommendations" in combined
    assert "begin with" not in combined
    assert "use this first" not in combined
    assert "call math.find first" not in combined
    assert "strongest one or two" not in combined
    assert "before searching for a checker" not in combined
    assert "partition larger searches" not in combined


def test_server_instructions_front_load_implicit_activation_signal() -> None:
    prefix = SERVER_INSTRUCTIONS[:512].lower()

    assert "specialized exact mathematical operation" in prefix
    assert "matrix or polynomial" in prefix
    assert "even when the user does not name jacobian" in prefix
    assert "math.find" in prefix


def test_server_instructions_allow_known_contracts_to_run_directly() -> None:
    assert "exact installed capability ID" in SERVER_INSTRUCTIONS
    assert "math.run may execute a known contract directly" in SERVER_INSTRUCTIONS


def test_server_instructions_route_pinned_lean_declaration_queries() -> None:
    assert (
        "explicitly targeting Jacobian's pinned CORE or MATHLIB" in SERVER_INSTRUCTIONS
    )
    assert "Project-local Lean declarations are outside" in SERVER_INSTRUCTIONS
    assert "may require project-local tools" in SERVER_INSTRUCTIONS


def test_guidance_rejects_verification_transfer_to_derived_claims() -> None:
    combined = "\n".join((SERVER_INSTRUCTIONS, MATH_RUN_DESCRIPTION))
    assert "does not verify a model-derived conclusion" in combined
    assert "record must be bound to the exact final claim" in combined
