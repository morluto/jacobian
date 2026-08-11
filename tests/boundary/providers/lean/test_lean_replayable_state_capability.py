from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityInvocationError
from jacobian.checker_authorization import LeanCheckerInstallation
from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import LeanProofStateRequest, LeanTypedGoal
from jacobian.lean_frontend.exploration import (
    _Resources,
    install_lean_exploration_capabilities,
)
from jacobian.lean_frontend.proof_state import LeanProofStateAdapter
from jacobian.lean_frontend.repl_protocol import (
    LeanReplCommandResponse,
    LeanReplErrorResponse,
    LeanReplProofResponse,
    LeanReplProofStepResponse,
    LeanReplValidatedExecution,
)
from jacobian.provider_runtime import jacobian_provider_runtime
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

_ReplResponses = LeanReplValidatedExecution


def _installation(environment: LeanEnvironment) -> LeanCheckerInstallation:
    digest = "artifact://sha256/" + (
        "a" * 64 if environment is LeanEnvironment.CORE else "b" * 64
    )
    return LeanCheckerInstallation(
        environment=environment,
        lean_version="4.31.0",
        lean_commit="lean-commit",
        import_name=None if environment is LeanEnvironment.CORE else "Mathlib",
        mathlib_commit=(
            None if environment is LeanEnvironment.CORE else "mathlib-commit"
        ),
        allowed_axioms=(),
        checker_timeout_seconds=30,
        semantics_uri=digest,
        claim_schema_uri=digest,
        candidate_schema_uri=digest,
        certificate_schema_uri=digest,
        checker_id="checker://sha256/" + "c" * 64,
    )


def _adapter(tmp_path: Path) -> LeanProofStateAdapter:
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    installations = {
        environment: _installation(environment) for environment in LeanEnvironment
    }
    adapters, _ = install_lean_exploration_capabilities(
        store,
        schemas,
        artifacts,
        installations,
        jacobian_provider_runtime(
            "jacobian.lean4",
            features=("clean-replay", "immutable-proof-state"),
        ),
    )
    return adapters[0]


def _responses(
    *,
    before: list[str],
    after: list[str],
    completed: bool = False,
    error: str | None = None,
) -> _ReplResponses:
    tactic: dict[str, object] = {
        "proofState": 2,
        "proofStatus": "Completed" if completed else "Goals",
        "goals": after,
    }
    if error is not None:
        tactic["messages"] = [
            {
                "pos": {"line": 0, "column": 0},
                "severity": "error",
                "data": error,
            }
        ]
    return (
        LeanReplCommandResponse.model_validate(
            {"env": 0, "sorries": [{"goal": "⊢ True", "proofState": 0}]}
        ),
        LeanReplProofStepResponse.model_validate(
            {"proofState": 1, "proofStatus": "Goals", "goals": before}
        ),
        LeanReplProofStepResponse.model_validate(tactic),
    )


def _stub_lean_runtime(
    monkeypatch: pytest.MonkeyPatch,
    adapter: LeanProofStateAdapter,
    responses: Callable[[], _ReplResponses],
) -> None:
    """Stub every Lean side channel used by apply_tactic.

    These tests intentionally stay off ``lean_runtime``: they validate adapter
    contracts with fake REPL payloads. Patching only ``execute_clean`` is an
    incomplete fixture — accepted tactics still call ``_extract_typed_goals``,
    which otherwise requires the pinned Lean helper binary.
    """

    last_goals: list[str] = []

    def _execute_clean(**kwargs: object) -> _ReplResponses:
        del kwargs
        payload = responses()
        tactic_response = payload[2]
        last_goals.clear()
        if isinstance(tactic_response, LeanReplProofStepResponse):
            last_goals.extend(tactic_response.goals)
        return payload

    def _fake_extract(
        _resources: _Resources,
        *,
        pickle_path: Path,
        request: LeanProofStateRequest,
    ) -> tuple[LeanTypedGoal, ...]:
        del _resources, pickle_path, request
        return tuple(
            LeanTypedGoal(
                goal_index=index,
                target_type=goal or "True",
                local_declarations=(),
            )
            for index, goal in enumerate(last_goals)
        )

    monkeypatch.setattr(adapter.resources.repl, "execute_clean", _execute_clean)
    monkeypatch.setattr(
        "jacobian.lean_frontend.exploration._extract_typed_goals",
        _fake_extract,
    )


def test_apply_tactic_materializes_and_reuses_replayable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(tmp_path)
    calls: Iterator[_ReplResponses] = iter(
        (
            _responses(
                before=["P Q : Prop  \r\n⊢ P ∧ Q"],
                after=["P Q : Prop\n⊢ P", "P Q : Prop\n⊢ Q"],
            ),
            _responses(
                before=["P Q : Prop\n⊢ P", "P Q : Prop\n⊢ Q"],
                after=[],
                completed=True,
            ),
        )
    )
    _stub_lean_runtime(monkeypatch, adapter, lambda: next(calls))

    first = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            input={
                "environment": "CORE",
                "statement": "(P Q : Prop) → P ∧ Q",
                "proof_prefix": ["intro P Q"],
                "tactic": "constructor",
            },
        )
    )
    successor_uri = first.output["successor_states"][0]["state_uri"]
    second = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            input={
                "environment": "CORE",
                "state_uri": successor_uri,
                "tactic": "all_goals assumption",
            },
        )
    )

    assert first.output["accepted"] is True
    assert first.output["completed"] is False
    assert first.output["goals"] == ["P Q : Prop\n⊢ P", "P Q : Prop\n⊢ Q"]
    assert first.output["input_state_uri"] in first.artifact_uris
    assert successor_uri in first.artifact_uris
    assert second.output["accepted"] is True
    assert second.output["completed"] is True
    assert second.output["successor_states"][0]["normalized_goals"] == []
    assert second.output["verification_boundary"] == "LEAN_CHECK_REQUIRED"
    assert second.output["verification"] == "UNVERIFIED"
    successor = adapter.resources.store.get(successor_uri)
    assert successor.payload["expiry"] == "IMMUTABLE_NO_EXPIRY"
    assert successor.payload["environment_digest"].startswith("sha256:")
    assert successor.payload["source_digest"].startswith("sha256:")
    assert successor.payload["state_digest"].startswith("sha256:")


def test_apply_tactic_returns_rejection_without_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(tmp_path)
    _stub_lean_runtime(
        monkeypatch,
        adapter,
        lambda: _responses(
            before=["P Q : Prop\nhP : P\n⊢ Q"],
            after=[],
            error="type mismatch: hP has type P",
        ),
    )

    result = adapter.invoke(
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
    assert result.output["completed"] is False
    assert result.output["successor_states"] == []
    assert result.output["diagnostics"][0]["severity"] == "ERROR"


@pytest.mark.parametrize(
    ("tactic_response", "expected_message"),
    (
        (
            LeanReplErrorResponse(message="tactic protocol error"),
            "tactic protocol error",
        ),
        (
            LeanReplProofStepResponse.model_validate(
                {"proofState": 2, "proofStatus": "failed", "goals": []}
            ),
            "Lean tactic returned proof status 'failed'",
        ),
    ),
)
def test_rejected_transition_persists_all_protocol_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tactic_response: LeanReplProofResponse,
    expected_message: str,
) -> None:
    adapter = _adapter(tmp_path)
    _stub_lean_runtime(
        monkeypatch,
        adapter,
        lambda: (
            LeanReplCommandResponse.model_validate(
                {"env": 0, "sorries": [{"goal": "⊢ True", "proofState": 0}]}
            ),
            LeanReplProofStepResponse.model_validate(
                {"proofState": 1, "proofStatus": "Goals", "goals": ["⊢ True"]}
            ),
            tactic_response,
        ),
    )

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            input={
                "environment": "CORE",
                "statement": "True",
                "tactic": "skip",
            },
        )
    )

    assert result.output["accepted"] is False
    assert expected_message in result.output["messages"]
    assert expected_message in {
        diagnostic["message"] for diagnostic in result.output["diagnostics"]
    }


def test_apply_tactic_rejects_environment_stale_state_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(tmp_path)
    _stub_lean_runtime(
        monkeypatch,
        adapter,
        lambda: _responses(before=["⊢ True"], after=[], completed=True),
    )
    opened = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            input={
                "environment": "CORE",
                "statement": "True",
                "tactic": "trivial",
            },
        )
    )
    state = adapter.resources.store.get(opened.output["input_state_uri"])
    stale_payload = dict(state.payload)
    stale_payload["environment_digest"] = "sha256:" + "f" * 64
    stale = adapter.resources.artifacts.put(
        schema_uri=adapter.resources.state_schema_uri,
        semantics_uri=adapter.resources.semantics_uri,
        payload=stale_payload,
        summary="fixture with stale environment binding",
    )

    with pytest.raises(CapabilityInvocationError) as raised:
        adapter.invoke(
            CapabilityRequest(
                capability_id="lean.proof_state.apply_tactic",
                input={
                    "environment": "CORE",
                    "state_uri": stale.artifact_uri,
                    "tactic": "trivial",
                },
            )
        )

    assert raised.value.diagnostic.code == "STALE_LEAN_PROOF_STATE"
