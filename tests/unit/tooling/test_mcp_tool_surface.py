"""Executable contract for the stable MCP tool surface."""

from __future__ import annotations

from jacobian.adapters.mcp.guidance import (
    MATH_FIND_DESCRIPTION,
    MATH_RUN_DESCRIPTION,
    SERVER_INSTRUCTIONS,
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

    assert "locally installed jacobian toolbox" in prefix
    assert "internet search cannot" in prefix
    assert "even when the user does not name jacobian" in prefix
    assert "math.find" in prefix


def test_server_instructions_distinguish_search_from_complete_inventory() -> None:
    combined = "\n".join((SERVER_INSTRUCTIONS, MATH_FIND_DESCRIPTION))

    assert "authoritative local search and exact-inspection interface" in combined
    assert "capability://catalog" in combined
    assert "math.find is the authoritative local inventory" not in combined
    assert "authoritative runtime inventory" not in combined


def test_server_instructions_allow_known_contracts_to_run_directly() -> None:
    assert "exact installed capability ID and its typed" in SERVER_INSTRUCTIONS
    assert "math.run may execute a known contract directly" in SERVER_INSTRUCTIONS
    assert "math.find is operation lookup, not confirmation" in SERVER_INSTRUCTIONS


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
    assert "model-authored duplicate calculations are not independent" in combined


def test_math_run_checks_status_before_using_output() -> None:
    status = MATH_RUN_DESCRIPTION.index("Check execution status")
    output = MATH_RUN_DESCRIPTION.index("treating `output`")
    assert status < output
    assert "operation-owned result fields" in MATH_RUN_DESCRIPTION
    assert "verification_record_uri" in MATH_RUN_DESCRIPTION
    assert "completeness" not in MATH_RUN_DESCRIPTION
    assert "assurance" not in MATH_RUN_DESCRIPTION


def test_math_run_copies_one_example_input_as_payload() -> None:
    assert "select one item from `invocation_examples`" in MATH_RUN_DESCRIPTION
    assert "item's `input` object as the `payload`" in MATH_RUN_DESCRIPTION
    assert "copy its `invocation_examples`" not in MATH_RUN_DESCRIPTION


def test_server_instructions_preserve_evidence_sensitive_abstention() -> None:
    assert "restating an accepted value without new evidence" in SERVER_INSTRUCTIONS
    assert "not a mathematical-tool use case" in SERVER_INSTRUCTIONS
