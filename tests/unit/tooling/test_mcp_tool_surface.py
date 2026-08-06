"""Executable contract for the stable MCP tool surface."""

from __future__ import annotations

from jacobian.adapters.mcp.constants import ReasoningLogMode
from jacobian.adapters.mcp.guidance import (
    MATH_FIND_DESCRIPTION,
    MATH_RUN_DESCRIPTION,
    SERVER_INSTRUCTIONS,
)
from jacobian.adapters.mcp.server import JacobianCoreExtension


def test_core_extension_exposes_exactly_the_stable_math_tools() -> None:
    extension = JacobianCoreExtension(None, None, ReasoningLogMode.OFF)
    assert extension.identifier == "io.jacobian/core"
    assert extension.settings() == {
        "version": "2",
        "reasoning_log_mode": "OFF",
    }
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
