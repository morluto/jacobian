from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.provider_lean import (
    PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
    PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    skip_unless_pinned_lean_core_runtime,
    skip_unless_pinned_mathlib_runtime,
)

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.runtime import CheckerAuthorityMode, create_runtime

pytestmark = [
    pytest.mark.skipif(
        skip_unless_pinned_lean_core_runtime(),
        reason=PINNED_LEAN_CORE_RUNTIME_UNAVAILABLE_REASON,
    ),
]


def test_apply_tactic_exposes_child_goals_and_replay_source(tmp_path: Path) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            input={
                "environment": "CORE",
                "statement": "(P Q : Prop) → P → Q → P ∧ Q",
                "proof_prefix": ["intro P Q hP hQ"],
                "tactic": "constructor",
            },
        )
    )

    assert result.output["completed"] is False
    assert result.output["goal_count"] == 2
    assert all("⊢" in goal for goal in result.output["goals"])
    assert [goal["target_type"] for goal in result.output["typed_goals"]] == [
        "P",
        "Q",
    ]
    assert result.output["typed_goals"][0]["local_declarations"] == [
        {"user_name": "P", "binder_info": "DEFAULT", "type": "Prop", "value": None},
        {"user_name": "Q", "binder_info": "DEFAULT", "type": "Prop", "value": None},
        {"user_name": "hP", "binder_info": "DEFAULT", "type": "P", "value": None},
        {"user_name": "hQ", "binder_info": "DEFAULT", "type": "Q", "value": None},
    ]
    assert result.output["accepted"] is True
    assert len(result.output["successor_states"]) == 1
    assert result.output["input_state_uri"] in result.artifact_uris
    assert result.output["successor_states"][0]["state_uri"] in result.artifact_uris
    assert result.output["transition_uri"] in result.artifact_uris
    assert result.output["replay_source"].endswith("intro P Q hP hQ\n  constructor")


def test_apply_tactic_returns_structured_failure_without_conclusion(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            input={
                "environment": "CORE",
                "statement": "(P Q : Prop) → P → Q",
                "proof_prefix": ["intro P Q hP"],
                "tactic": "exact hP",
            },
        )
    )

    assert result.execution.status.value == "COMPLETED"
    assert result.output["accepted"] is False
    assert result.output["successor_states"] == []
    assert any(
        diagnostic["severity"] == "ERROR" for diagnostic in result.output["diagnostics"]
    )


@pytest.mark.skipif(
    skip_unless_pinned_mathlib_runtime(),
    reason=PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
)
def test_retrieve_premises_returns_exact_mathlib_suggestion(tmp_path: Path) -> None:
    runtime = create_runtime(
        tmp_path, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    )

    suggested = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.retrieve.premises",
            input={
                "environment": "MATHLIB",
                "statement": "(n : Nat) → Nat.gcd n 0 = n",
                "proof_prefix": ["intro n"],
                "limit": 5,
            },
        )
    )

    empty = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.retrieve.premises",
            input={
                "environment": "MATHLIB",
                "statement": "(P Q : Prop) → P → Q",
                "proof_prefix": ["intro P Q hP"],
                "limit": 5,
            },
        )
    )

    assert suggested.output["candidates"]
    assert suggested.output["candidates"][0]["tactic"] == ("exact Nat.gcd_zero_right n")
    assert (
        "Nat.gcd_zero_right" in suggested.output["candidates"][0]["declaration_names"]
    )
    assert suggested.output["candidates"][0]["tactic_replayed"] is True
    assert (
        suggested.output["candidates"][0]["declaration_name_extraction"]
        == "DISPLAY_TEXT_HEURISTIC"
    )
    assert suggested.output["api_stability"] == "EXPERIMENTAL_TACTIC_DIAGNOSTIC"
    assert suggested.output["goal_context_digest"].startswith("sha256:")
    assert suggested.output["retrieval_uri"] in suggested.artifact_uris
    assert empty.execution.status.value == "COMPLETED"
    assert empty.output["candidates"] == []
    assert empty.output["exhaustive"] is False


def test_runtime_can_ablate_lean_capabilities_without_removing_checker(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(
        tmp_path,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
        capability_exclusions=frozenset(
            {
                "lean.proof_state.apply_tactic",
                "lean.retrieve.premises",
            }
        ),
    )
    capability_ids = {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }

    assert "lean.check" in capability_ids
    assert "lean.proof_state.apply_tactic" not in capability_ids
    assert "lean.retrieve.premises" not in capability_ids
