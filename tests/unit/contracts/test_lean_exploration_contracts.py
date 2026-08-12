from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.lean import (
    LeanDiagnostic,
    LeanDiagnosticPhase,
    LeanDiagnosticSource,
    LeanDiagnosticSourceSpan,
)
from jacobian.contracts.lean_exploration import (
    LeanPremiseRetrievalRequest,
    LeanProofStateRequest,
    LeanProofStateTransitionArtifact,
)


def test_proof_state_request_has_hard_structured_output_bounds() -> None:
    with pytest.raises(ValidationError):
        LeanProofStateRequest(statement="True", tactic="trivial", max_goals=65)
    with pytest.raises(ValidationError):
        LeanProofStateRequest(
            statement="True",
            tactic="trivial",
            max_local_declarations=257,
        )


@pytest.mark.parametrize(
    "request_model",
    [
        LeanProofStateRequest,
        LeanPremiseRetrievalRequest,
    ],
)
def test_proof_prefix_is_the_tactic_body_after_by(request_model: type[object]) -> None:
    payload = {
        "statement": "True",
        "proof_prefix": ["by", "trivial"],
    }
    if request_model is LeanProofStateRequest:
        payload["tactic"] = "trivial"

    with pytest.raises(ValidationError, match="must not include `by`"):
        request_model.model_validate(payload)  # type: ignore[attr-defined]


def test_lean_exploration_request_schemas_explain_fresh_and_continuation_modes() -> (
    None
):
    state_schema = LeanProofStateRequest.model_json_schema()["properties"]
    premise_schema = LeanPremiseRetrievalRequest.model_json_schema()["properties"]

    assert "continuation" in state_schema["state_uri"]["description"].lower()
    assert "fresh" in state_schema["statement"]["description"].lower()
    assert "Do not include `by`" in state_schema["proof_prefix"]["description"]
    assert "Do not include `by`" in premise_schema["proof_prefix"]["description"]


def test_transition_binds_rendered_and_typed_goal_counts() -> None:
    with pytest.raises(ValidationError, match="typed goals"):
        LeanProofStateTransitionArtifact(
            environment="CORE",
            environment_digest="sha256:" + "a" * 64,
            source_digest="sha256:" + "b" * 64,
            statement="True",
            proof_prefix=(),
            tactic="skip",
            input_state_uri="artifact://sha256/" + "c" * 64,
            input_state_digest="sha256:" + "d" * 64,
            replay_source="skip",
            goals=("⊢ True",),
            typed_goals=(),
            goal_count=1,
            successor_states=(),
            accepted=True,
            completed=False,
            messages=(),
            diagnostics=(),
            lean_version="4.31.0",
            lean_commit="abc",
        )


def test_lean_diagnostic_carries_machine_actionable_payload_location() -> None:
    diagnostic = LeanDiagnostic(
        code="LEAN_TYPE_MISMATCH",
        phase=LeanDiagnosticPhase.TERM_ELABORATION,
        severity="ERROR",
        message="The supplied term has the wrong type.",
        source_span=LeanDiagnosticSourceSpan(
            source=LeanDiagnosticSource.TERM,
            start={"line": 0, "column": 0},
            end={"line": 0, "column": 7},
        ),
        goal_index=0,
        metavariable="?m.42",
        raw_backend_message="type mismatch: trivial has type True",
    )

    assert diagnostic.code == "LEAN_TYPE_MISMATCH"
    assert diagnostic.source_span is not None
    assert diagnostic.source_span.source is LeanDiagnosticSource.TERM
    assert diagnostic.goal_index == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"code": "type-mismatch"},
        {"goal_index": -1},
        {
            "source_span": {
                "source": "TACTIC",
                "start": {"line": 1, "column": 0},
                "end": {"line": 0, "column": 1},
            }
        },
    ],
)
def test_lean_diagnostic_rejects_unstable_or_invalid_coordinates(
    payload: dict[str, object],
) -> None:
    base: dict[str, object] = {
        "code": "LEAN_TACTIC_REJECTED",
        "phase": "TACTIC_EXECUTION",
        "severity": "ERROR",
        "message": "Lean rejected the tactic.",
        "raw_backend_message": "tactic failed",
    }
    base.update(payload)

    with pytest.raises(ValidationError):
        LeanDiagnostic.model_validate(base)
