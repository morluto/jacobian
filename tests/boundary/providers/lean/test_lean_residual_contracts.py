"""Contract-level tests for the #95 Lean residual contracts.

These tests validate the three bounded contracts without starting a real Lean
process: the REPL transport and the pinned typed-goal helper are stubbed so
the adapter contracts, artifact shapes, fail-closed behavior, and the honest
coercion-provenance limitation can be checked on any host.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import LeanProofStateRequest, LeanTypedGoal
from jacobian.lean_frontend.exploration import (
    _parse_typed_goal_envelope as _parse_typed_goal_envelope_public,
)
from jacobian.lean_frontend.exploration import (
    _Resources,
    install_lean_exploration_capabilities,
)
from jacobian.lean_frontend.helper_protocol import (
    LeanMetavariableFieldsHelperPayload,
)
from jacobian.lean_frontend.metavariable_fields import LeanMetavariableFieldsAdapter
from jacobian.lean_frontend.proof_state import LeanProofStateAdapter
from jacobian.lean_frontend.proof_state_inspect import LeanProofStateInspectAdapter
from jacobian.lean_frontend.repl_protocol import (
    LeanReplCommandResponse,
    LeanReplProofStepResponse,
    LeanReplValidatedExecution,
)
from jacobian.lean_frontend.term_apply import LeanTermApplyAdapter
from jacobian.provider_runtime import jacobian_provider_runtime
from jacobian.references import LeanCheckerInstallation
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


def _adapters(
    tmp_path: Path,
) -> tuple[
    LeanProofStateAdapter,
    LeanTermApplyAdapter,
    LeanProofStateInspectAdapter,
    LeanMetavariableFieldsAdapter,
    _Resources,
]:
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    installations = {
        environment: _installation(environment) for environment in LeanEnvironment
    }
    all_adapters, _ = install_lean_exploration_capabilities(
        store,
        schemas,
        artifacts,
        installations,
        jacobian_provider_runtime(
            "jacobian.lean4",
            features=(
                "clean-replay",
                "immutable-proof-state",
                "term-apply",
                "metavariable-fields",
            ),
        ),
    )
    resources = all_adapters[0].resources
    return (
        all_adapters[0],
        all_adapters[2],
        all_adapters[3],
        all_adapters[4],
        resources,
    )


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


def _stub_apply_runtime(
    monkeypatch: pytest.MonkeyPatch,
    adapter: LeanProofStateAdapter,
    responses: Callable[[], _ReplResponses],
) -> None:
    last_goals: list[str] = []

    def _execute_clean(**kwargs: object) -> _ReplResponses:
        del kwargs
        payload = responses()
        tactic_response = payload[2]
        if not isinstance(tactic_response, LeanReplProofStepResponse):
            raise TypeError("stub tactic response must be a proof response")
        last_goals.clear()
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


def _stub_metavariable_runtime(
    monkeypatch: pytest.MonkeyPatch,
    adapter: LeanMetavariableFieldsAdapter,
    *,
    structured: list[dict[str, Any]],
    elaboration: dict[str, Any],
    before: list[str],
) -> None:
    def _execute_clean(**kwargs: object) -> _ReplResponses:
        del kwargs
        return _responses(before=before, after=before)

    def _fake_extract_structured(
        _resources: _Resources,
        *,
        pickle_path: Path,
        request: Any,
    ) -> LeanMetavariableFieldsHelperPayload:
        del _resources, pickle_path, request
        return LeanMetavariableFieldsHelperPayload.model_validate(
            {
                "expression_serialization": "LEAN_PRETTY_PRINTED_EXPR",
                "structured_metavariables": structured,
                "elaboration_context": elaboration,
                "coercion_provenance": "UNAVAILABLE",
                "coercion_provenance_basis": (
                    "maintained Lean.Meta.Coe APIs operate on expressions during "
                    "elaboration; a pickled proof state retains no per-metavariable "
                    "coercion log"
                ),
            }
        )

    monkeypatch.setattr(adapter.resources.repl, "execute_clean", _execute_clean)
    monkeypatch.setattr(
        "jacobian.lean_frontend.exploration._extract_structured_metavariables",
        _fake_extract_structured,
    )


def _stored_input_state_uri(
    monkeypatch: pytest.MonkeyPatch,
    proof_state: LeanProofStateAdapter,
    environment: LeanEnvironment,
) -> str:
    _stub_apply_runtime(
        monkeypatch,
        proof_state,
        lambda: _responses(before=["⊢ True"], after=["⊢ True"]),
    )
    opened = proof_state.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": environment.value,
                "statement": "True",
                "tactic": "skip",
            },
        )
    )
    return str(opened.output["input_state_uri"])


def _invoke_stored_state_consumer(
    consumer: str,
    *,
    proof_state: LeanProofStateAdapter,
    inspect: LeanProofStateInspectAdapter,
    metavariable: LeanMetavariableFieldsAdapter,
    environment: LeanEnvironment,
    state_uri: str,
) -> None:
    request_input: dict[str, object] = {
        "environment": environment.value,
        "state_uri": state_uri,
    }
    if consumer == "apply_tactic":
        adapter = proof_state
        capability_id = "lean.proof_state.apply_tactic"
        request_input["tactic"] = "skip"
    elif consumer == "inspect":
        adapter = inspect
        capability_id = "lean.proof_state.inspect"
    else:
        adapter = metavariable
        capability_id = "lean.proof_state.metavariable_fields"
    adapter.invoke(
        CapabilityRequest(
            capability_id=capability_id,
            mode=CapabilityMode.EXPLORE,
            input=request_input,
        )
    )


# ---------------------------------------------------------------------------
# lean.term.apply
# ---------------------------------------------------------------------------


def test_term_apply_elaborates_exact_term_and_returns_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_state, term_apply, _, _, _ = _adapters(tmp_path)
    calls: Iterator[_ReplResponses] = iter(
        (
            _responses(
                before=["P : Prop\n⊢ P"],
                after=[],
                completed=True,
            ),
        )
    )
    _stub_apply_runtime(monkeypatch, proof_state, lambda: next(calls))

    result = term_apply.invoke(
        CapabilityRequest(
            capability_id="lean.term.apply",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "statement": "P → P",
                "proof_prefix": ["intro P"],
                "term": "P",
            },
        )
    )

    assert result.capability_id == "lean.term.apply"
    assert result.output["accepted"] is True
    assert result.output["completed"] is True
    assert result.output["tactic"] == "exact P"
    assert result.output["term_application"] == "LEAN_EXACT_ELABORATION"
    assert result.output["term_apply_uri"] == result.output["transition_uri"]
    assert result.output["verification_boundary"] == "LEAN_CHECK_REQUIRED"
    assert result.output["verification"] == "UNVERIFIED"
    assert result.output["successor_states"][0]["normalized_goals"] == []


def test_term_apply_rejects_multiline_term(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, term_apply, _, _, _ = _adapters(tmp_path)
    with pytest.raises(CapabilityInvocationError) as raised:
        term_apply.invoke(
            CapabilityRequest(
                capability_id="lean.term.apply",
                mode=CapabilityMode.EXPLORE,
                input={
                    "environment": "CORE",
                    "statement": "True",
                    "term": "trivial\nsorry",
                },
            )
        )
    assert raised.value.diagnostic.code == "INVALID_LEAN_TERM_APPLY_REQUEST"


def test_term_apply_fails_closed_on_rejected_term(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_state, term_apply, _, _, _ = _adapters(tmp_path)
    _stub_apply_runtime(
        monkeypatch,
        proof_state,
        lambda: _responses(
            before=["n : Nat\n⊢ n = 0"],
            after=[],
            error="type mismatch: trivial has type True",
        ),
    )
    result = term_apply.invoke(
        CapabilityRequest(
            capability_id="lean.term.apply",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "statement": "n = 0",
                "proof_prefix": ["intro n"],
                "term": "trivial",
            },
        )
    )
    assert result.output["accepted"] is False
    assert result.output["successor_states"] == []
    assert result.output["term_application"] == "LEAN_EXACT_ELABORATION"


# ---------------------------------------------------------------------------
# lean.proof_state.inspect
# ---------------------------------------------------------------------------


def test_inspect_returns_recorded_goals_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_state, _, inspect, _, _ = _adapters(tmp_path)
    _stub_apply_runtime(
        monkeypatch,
        proof_state,
        lambda: _responses(
            before=["P Q : Prop\n⊢ P ∧ Q"],
            after=["P Q : Prop\n⊢ P", "P Q : Prop\n⊢ Q"],
        ),
    )
    opened = proof_state.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "statement": "(P Q : Prop) → P ∧ Q",
                "proof_prefix": ["intro P Q"],
                "tactic": "constructor",
            },
        )
    )
    successor_uri = opened.output["successor_states"][0]["state_uri"]

    result = inspect.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.inspect",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "state_uri": successor_uri,
            },
        )
    )

    assert result.capability_id == "lean.proof_state.inspect"
    assert result.output["inspection"] == "READ_ONLY_NO_REPLAY"
    assert result.output["verification"] == "UNVERIFIED"
    assert result.output["completed"] is False
    assert result.output["goal_count"] == 2
    assert result.output["normalized_goals"] == [
        "P Q : Prop\n⊢ P",
        "P Q : Prop\n⊢ Q",
    ]
    assert result.output["state_uri"] == successor_uri
    assert result.artifact_uris == (successor_uri,)


def test_inspect_rejects_stale_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_state, _, inspect, _, resources = _adapters(tmp_path)
    _stub_apply_runtime(
        monkeypatch,
        proof_state,
        lambda: _responses(before=["⊢ True"], after=["⊢ True"]),
    )
    opened = proof_state.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            mode=CapabilityMode.EXPLORE,
            input={"environment": "CORE", "statement": "True", "tactic": "skip"},
        )
    )
    state = resources.store.get(opened.output["input_state_uri"])
    stale_payload = dict(state.payload)
    stale_payload["environment_digest"] = "sha256:" + "f" * 64
    stale = resources.artifacts.put(
        schema_uri=resources.state_schema_uri,
        semantics_uri=resources.semantics_uri,
        payload=stale_payload,
        summary="fixture with stale environment binding",
    )
    with pytest.raises(CapabilityInvocationError) as raised:
        inspect.invoke(
            CapabilityRequest(
                capability_id="lean.proof_state.inspect",
                mode=CapabilityMode.EXPLORE,
                input={
                    "environment": "CORE",
                    "state_uri": stale.artifact_uri,
                },
            )
        )
    assert raised.value.diagnostic.code == "STALE_LEAN_PROOF_STATE"


# ---------------------------------------------------------------------------
# lean.proof_state.metavariable_fields
# ---------------------------------------------------------------------------


def _metavariable_fixture() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    structured = [
        {
            "goal_index": 0,
            "user_name": "_anon",
            "is_user_name_anonymous": True,
            "kind": "NATURAL",
            "is_assigned": False,
            "is_delayed_assigned": False,
            "depth": 1,
            "num_scope_args": 0,
            "target_type": "Prop",
            "local_instances": [
                {
                    "class_name": "Decidable",
                    "fvar_user_name": "inst",
                    "fvar_type": "Decidable P",
                }
            ],
        }
    ]
    elaboration = {
        "decl_name": "",
        "may_postpone": True,
        "err_to_sorry": True,
        "auto_bound_implicit": False,
        "implicit_lambda": True,
        "is_noncomputable_section": False,
        "ignore_tc_failures": False,
        "in_pattern": False,
        "save_rec_app_syntax": True,
        "holes_as_synthetic_opaque": False,
    }
    return structured, elaboration


def test_metavariable_fields_expose_structured_fields_and_unavailable_coercion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_state, _, _, metavariable, _ = _adapters(tmp_path)
    _stub_apply_runtime(
        monkeypatch,
        proof_state,
        lambda: _responses(before=["P : Prop\n⊢ P"], after=["P : Prop\n⊢ P"]),
    )
    opened = proof_state.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "statement": "P → P",
                "proof_prefix": ["intro P"],
                "tactic": "skip",
            },
        )
    )
    state_uri = opened.output["successor_states"][0]["state_uri"]
    structured, elaboration = _metavariable_fixture()
    _stub_metavariable_runtime(
        monkeypatch,
        metavariable,
        structured=structured,
        elaboration=elaboration,
        before=["P : Prop\n⊢ P"],
    )

    result = metavariable.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.metavariable_fields",
            mode=CapabilityMode.EXPLORE,
            input={"environment": "CORE", "state_uri": state_uri},
        )
    )

    assert result.capability_id == "lean.proof_state.metavariable_fields"
    assert result.output["metavariable_schema_version"] == "1"
    assert result.output["coercion_provenance"] == "UNAVAILABLE"
    assert "Lean.Meta.Coe" in result.output["coercion_provenance_basis"]
    assert result.output["verification"] == "UNVERIFIED"
    mvar = result.output["structured_metavariables"][0]
    assert mvar["goal_index"] == 0
    assert mvar["kind"] == "NATURAL"
    assert mvar["is_assigned"] is False
    assert mvar["local_instances"][0]["class_name"] == "Decidable"
    elab = result.output["elaboration_context"]
    assert elab["may_postpone"] is True
    assert elab["holes_as_synthetic_opaque"] is False
    assert state_uri in result.artifact_uris


def test_metavariable_fields_reject_completed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_state, _, _, metavariable, _ = _adapters(tmp_path)
    _stub_apply_runtime(
        monkeypatch,
        proof_state,
        lambda: _responses(before=["⊢ True"], after=[], completed=True),
    )
    opened = proof_state.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            mode=CapabilityMode.EXPLORE,
            input={"environment": "CORE", "statement": "True", "tactic": "trivial"},
        )
    )
    state_uri = opened.output["successor_states"][0]["state_uri"]
    with pytest.raises(CapabilityInvocationError) as raised:
        metavariable.invoke(
            CapabilityRequest(
                capability_id="lean.proof_state.metavariable_fields",
                mode=CapabilityMode.EXPLORE,
                input={"environment": "CORE", "state_uri": state_uri},
            )
        )
    assert raised.value.diagnostic.code == "LEAN_PROOF_STATE_COMPLETED"


def test_metavariable_fields_fails_closed_on_helper_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_state, _, _, metavariable, _ = _adapters(tmp_path)
    _stub_apply_runtime(
        monkeypatch,
        proof_state,
        lambda: _responses(before=["P : Prop\n⊢ P"], after=["P : Prop\n⊢ P"]),
    )
    opened = proof_state.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "statement": "P → P",
                "proof_prefix": ["intro P"],
                "tactic": "skip",
            },
        )
    )
    state_uri = opened.output["successor_states"][0]["state_uri"]

    def _execute_clean(**kwargs: object) -> _ReplResponses:
        del kwargs
        return _responses(before=["P : Prop\n⊢ P"], after=["P : Prop\n⊢ P"])

    def _fail(*_a: object, **_k: object) -> dict[str, Any]:
        raise RuntimeError("helper protocol failed")

    monkeypatch.setattr(metavariable.resources.repl, "execute_clean", _execute_clean)
    monkeypatch.setattr(
        "jacobian.lean_frontend.exploration._extract_structured_metavariables",
        _fail,
    )
    with pytest.raises(CapabilityInvocationError) as raised:
        metavariable.invoke(
            CapabilityRequest(
                capability_id="lean.proof_state.metavariable_fields",
                mode=CapabilityMode.EXPLORE,
                input={"environment": "CORE", "state_uri": state_uri},
            )
        )
    assert raised.value.diagnostic.code == "LEAN_METAVARIABLE_FIELDS_EXTRACTION_FAILED"


# ---------------------------------------------------------------------------
# contract model validation
# ---------------------------------------------------------------------------


def test_metavariable_artifact_rejects_non_contiguous_indices() -> None:
    from jacobian.contracts.lean_metavariable_fields import (
        LeanElaborationContext,
        LeanMetavariableFieldsArtifact,
        LeanStructuredMetavariable,
    )

    digest = "sha256:" + "a" * 64
    artifact_uri = "artifact://sha256/" + "a" * 64
    mvar = LeanStructuredMetavariable(
        goal_index=1,
        user_name="x",
        is_user_name_anonymous=False,
        kind="NATURAL",
        is_assigned=False,
        is_delayed_assigned=False,
        depth=0,
        num_scope_args=0,
        target_type="Prop",
        local_instances=(),
    )
    elab = LeanElaborationContext(
        decl_name="",
        may_postpone=True,
        err_to_sorry=True,
        auto_bound_implicit=False,
        implicit_lambda=True,
        is_noncomputable_section=False,
        ignore_tc_failures=False,
        in_pattern=False,
        save_rec_app_syntax=True,
        holes_as_synthetic_opaque=False,
    )
    with pytest.raises(ValueError, match="contiguous"):
        LeanMetavariableFieldsArtifact(
            environment=LeanEnvironment.CORE,
            environment_digest=digest,
            source_digest=digest,
            state_uri=artifact_uri,
            state_digest=digest,
            structured_metavariables=(mvar,),
            elaboration_context=elab,
            coercion_provenance_basis="unavailable",
            lean_version="4.31.0",
            lean_commit="c",
        )


def test_term_apply_request_requires_one_state_source() -> None:
    from jacobian.contracts.lean_term_apply import LeanTermApplyRequest

    with pytest.raises(ValueError, match="statement is required"):
        LeanTermApplyRequest.model_validate({"term": "trivial"})
    with pytest.raises(ValueError, match="state_uri cannot be combined"):
        LeanTermApplyRequest.model_validate(
            {
                "state_uri": "artifact://sha256/" + "a" * 64,
                "statement": "True",
                "term": "trivial",
            }
        )


# ---------------------------------------------------------------------------
# regression: forbidden term rejected at term_apply boundary (M3 fix)
# ---------------------------------------------------------------------------


def test_term_apply_rejects_sorry_at_own_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, term_apply, _, _, _ = _adapters(tmp_path)
    with pytest.raises(CapabilityInvocationError) as raised:
        term_apply.invoke(
            CapabilityRequest(
                capability_id="lean.term.apply",
                mode=CapabilityMode.EXPLORE,
                input={
                    "environment": "CORE",
                    "statement": "True",
                    "term": "sorry",
                },
            )
        )
    assert raised.value.diagnostic.code == "INVALID_LEAN_TERM_APPLY_REQUEST"
    assert raised.value.diagnostic.stage == "request_validation"


def test_term_apply_rejects_admit_at_own_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, term_apply, _, _, _ = _adapters(tmp_path)
    with pytest.raises(CapabilityInvocationError) as raised:
        term_apply.invoke(
            CapabilityRequest(
                capability_id="lean.term.apply",
                mode=CapabilityMode.EXPLORE,
                input={
                    "environment": "CORE",
                    "statement": "True",
                    "term": "admit",
                },
            )
        )
    assert raised.value.diagnostic.code == "INVALID_LEAN_TERM_APPLY_REQUEST"


# ---------------------------------------------------------------------------
# regression: helper error envelopes preserved (M2 fix)
# ---------------------------------------------------------------------------


def test_helper_error_envelope_preserves_specific_code() -> None:
    from jacobian.lean_frontend.exploration import LeanHelperError

    # Simulate helper stderr that emits an error envelope with a specific code
    error_output = (
        b"JACOBIAN_PROOF_STATE_ERROR "
        b'{"request_id":"abc","code":"LEAN_PROOF_STATE_GOAL_LIMIT",'
        b'"message":"typed proof-state extraction failed"}\n'
    )
    with pytest.raises(LeanHelperError) as raised:
        _parse_typed_goal_envelope_public(error_output, request_id="abc")
    assert raised.value.code == "LEAN_PROOF_STATE_GOAL_LIMIT"


def test_helper_error_envelope_preserves_unknown_mode_code() -> None:
    from jacobian.lean_frontend.exploration import LeanHelperError

    error_output = (
        b"JACOBIAN_PROOF_STATE_ERROR "
        b'{"request_id":"xyz","code":"LEAN_PROOF_STATE_UNKNOWN_MODE",'
        b'"message":"typed proof-state extraction failed"}\n'
    )
    with pytest.raises(LeanHelperError) as raised:
        _parse_typed_goal_envelope_public(error_output, request_id="xyz")
    assert raised.value.code == "LEAN_PROOF_STATE_UNKNOWN_MODE"


def test_helper_error_envelope_rejects_mismatched_request_id() -> None:
    from jacobian.lean_frontend.exploration import LeanHelperError

    error_output = (
        b"JACOBIAN_PROOF_STATE_ERROR "
        b'{"request_id":"other","code":"LEAN_PROOF_STATE_GOAL_LIMIT",'
        b'"message":"typed proof-state extraction failed"}\n'
    )
    # A mismatched request_id must NOT surface the specific code; it falls
    # through to the generic RuntimeError so a stale/desynced envelope is
    # never trusted.
    with pytest.raises(RuntimeError) as raised:
        _parse_typed_goal_envelope_public(error_output, request_id="abc")
    assert not isinstance(raised.value, LeanHelperError)


def test_helper_result_envelope_still_parses_normally() -> None:
    payload = _parse_typed_goal_envelope_public(
        b"JACOBIAN_PROOF_STATE_RESULT "
        b'{"request_id":"req1","payload":{"expression_serialization":'
        b'"LEAN_PRETTY_PRINTED_EXPR","typed_goals":[]}}\n',
        request_id="req1",
    )
    assert payload.expression_serialization == "LEAN_PRETTY_PRINTED_EXPR"


# ---------------------------------------------------------------------------
# regression: inspect available without Lean runtime (comment F)
# ---------------------------------------------------------------------------


def test_inspect_adapter_available_without_lean_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.lean_frontend.proof_state_inspect as inspect_module
    from jacobian.lean_frontend.proof_state_inspect import (
        install_lean_proof_state_inspect_only,
    )
    from jacobian.provider_runtime import jacobian_provider_runtime

    def _unexpected_runtime(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("read-only inspection constructed a Lean runtime")

    monkeypatch.setattr(
        inspect_module,
        "LeanExplorationReplRuntime",
        _unexpected_runtime,
        raising=False,
    )

    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    installations = {
        environment: _installation(environment) for environment in LeanEnvironment
    }
    adapter = install_lean_proof_state_inspect_only(
        store,
        schemas,
        artifacts,
        installations,
        jacobian_provider_runtime(
            "jacobian.lean4",
            features=("immutable-proof-state",),
        ),
    )

    from jacobian.lean_frontend.artifacts import (
        _environment_digest,
        _state_payload,
    )

    installation = installations[LeanEnvironment.CORE]
    state = artifacts.put(
        schema_uri=adapter.resources.state_schema_uri,
        semantics_uri=adapter.resources.semantics_uri,
        payload=_state_payload(
            environment=LeanEnvironment.CORE,
            environment_digest=_environment_digest(
                LeanEnvironment.CORE,
                installation,
            ),
            statement="True",
            tactic_prefix=(),
            normalized_goals=("⊢ True",),
            installation=installation,
        ).model_dump(mode="json"),
        summary="stored CORE proof state",
    )
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.inspect",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "state_uri": state.artifact_uri,
            },
        )
    )

    assert adapter.descriptor.capability_id == "lean.proof_state.inspect"
    assert adapter.descriptor.read_only is True
    assert result.output["normalized_goals"] == ["⊢ True"]
    assert (
        result.execution.detail == "read-only inspection; no Lean process was started"
    )


@pytest.mark.parametrize(
    ("stored_environment", "requested_environment"),
    (
        (LeanEnvironment.CORE, LeanEnvironment.MATHLIB),
        (LeanEnvironment.MATHLIB, LeanEnvironment.CORE),
    ),
)
@pytest.mark.parametrize(
    "consumer",
    ("apply_tactic", "inspect", "metavariable_fields"),
)
def test_stored_state_consumers_reject_cross_profile_artifacts_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored_environment: LeanEnvironment,
    requested_environment: LeanEnvironment,
    consumer: str,
) -> None:
    proof_state, _, inspect, metavariable, resources = _adapters(tmp_path)
    state_uri = _stored_input_state_uri(
        monkeypatch,
        proof_state,
        stored_environment,
    )

    def _unexpected_replay(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("cross-profile state reached the Lean runtime")

    monkeypatch.setattr(resources.repl, "execute_clean", _unexpected_replay)
    with pytest.raises(CapabilityInvocationError) as raised:
        _invoke_stored_state_consumer(
            consumer,
            proof_state=proof_state,
            inspect=inspect,
            metavariable=metavariable,
            environment=requested_environment,
            state_uri=state_uri,
        )

    assert raised.value.diagnostic.code == "STALE_LEAN_PROOF_STATE"
    assert raised.value.diagnostic.stage == "state_validation"


@pytest.mark.parametrize(
    ("consumer", "expected_hint"),
    (
        ("apply_tactic", "Use a state URI returned by this capability."),
        (
            "inspect",
            "Use a state URI returned by a proof-state capability.",
        ),
        (
            "metavariable_fields",
            "Use a state URI returned by a proof-state capability.",
        ),
    ),
)
def test_invalid_state_diagnostics_keep_consumer_specific_hints(
    tmp_path: Path,
    consumer: str,
    expected_hint: str,
) -> None:
    proof_state, _, inspect, metavariable, _ = _adapters(tmp_path)
    missing_uri = "artifact://sha256/" + "d" * 64
    with pytest.raises(CapabilityInvocationError) as raised:
        _invoke_stored_state_consumer(
            consumer,
            proof_state=proof_state,
            inspect=inspect,
            metavariable=metavariable,
            environment=LeanEnvironment.CORE,
            state_uri=missing_uri,
        )

    assert raised.value.diagnostic.code == "INVALID_LEAN_PROOF_STATE"
    assert raised.value.diagnostic.hint == expected_hint


@pytest.mark.parametrize("stale_binding", ("source_digest", "state_digest"))
def test_metavariable_fields_rejects_stale_state_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_binding: str,
) -> None:
    proof_state, _, _, metavariable, resources = _adapters(tmp_path)
    state_uri = _stored_input_state_uri(
        monkeypatch,
        proof_state,
        LeanEnvironment.CORE,
    )
    stored = resources.store.get(state_uri)
    stale_payload = dict(stored.payload)
    stale_payload[stale_binding] = "sha256:" + "f" * 64
    if stale_binding == "source_digest":
        from jacobian.contracts.lean_exploration import LeanProofStateArtifact
        from jacobian.lean_frontend.artifacts import _state_digest_payload

        stale_payload["state_digest"] = _state_digest_payload(
            LeanProofStateArtifact.model_validate(stale_payload)
        )
    stale = resources.artifacts.put(
        schema_uri=resources.state_schema_uri,
        semantics_uri=resources.semantics_uri,
        payload=stale_payload,
        summary=f"fixture with stale {stale_binding}",
    )

    def _unexpected_replay(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("stale state reached the Lean runtime")

    monkeypatch.setattr(resources.repl, "execute_clean", _unexpected_replay)
    with pytest.raises(CapabilityInvocationError) as raised:
        metavariable.invoke(
            CapabilityRequest(
                capability_id="lean.proof_state.metavariable_fields",
                mode=CapabilityMode.EXPLORE,
                input={
                    "environment": "CORE",
                    "state_uri": stale.artifact_uri,
                },
            )
        )

    assert raised.value.diagnostic.code == "STALE_LEAN_PROOF_STATE"


# ---------------------------------------------------------------------------
# regression: inspect rejects forged environment metadata (comment H)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("environment", "field", "forged_value"),
    (
        (LeanEnvironment.CORE, "imports", ["Mathlib"]),
        (LeanEnvironment.MATHLIB, "imports", ["Init"]),
        (LeanEnvironment.CORE, "lean_version", "9.9.9"),
        (LeanEnvironment.CORE, "lean_commit", "forged-lean-commit"),
        (LeanEnvironment.MATHLIB, "mathlib_commit", "forged-mathlib-commit"),
    ),
)
def test_inspect_rejects_forged_environment_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: LeanEnvironment,
    field: str,
    forged_value: object,
) -> None:
    proof_state, _, inspect, _, resources = _adapters(tmp_path)
    state_uri = _stored_input_state_uri(
        monkeypatch,
        proof_state,
        environment,
    )
    state = resources.store.get(state_uri)
    forged_payload = dict(state.payload)
    forged_payload[field] = forged_value
    # Recompute state_digest so the digest check alone would pass.
    from jacobian.contracts.lean_exploration import LeanProofStateArtifact
    from jacobian.lean_frontend.artifacts import _state_digest_payload

    forged_artifact = LeanProofStateArtifact.model_validate(forged_payload)
    forged_payload["state_digest"] = _state_digest_payload(forged_artifact)
    forged = resources.artifacts.put(
        schema_uri=resources.state_schema_uri,
        semantics_uri=resources.semantics_uri,
        payload=forged_payload,
        summary="fixture with forged lean_version",
    )
    with pytest.raises(CapabilityInvocationError) as raised:
        inspect.invoke(
            CapabilityRequest(
                capability_id="lean.proof_state.inspect",
                mode=CapabilityMode.EXPLORE,
                input={
                    "environment": environment.value,
                    "state_uri": forged.artifact_uri,
                },
            )
        )
    assert raised.value.diagnostic.code == "STALE_LEAN_PROOF_STATE"


# ---------------------------------------------------------------------------
# regression: term apply validates output via LeanTermApplyOutput (comment I)
# ---------------------------------------------------------------------------


def test_term_apply_output_is_validated_through_typed_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_state, term_apply, _, _, _ = _adapters(tmp_path)
    _stub_apply_runtime(
        monkeypatch,
        proof_state,
        lambda: _responses(
            before=["P : Prop\n⊢ P"],
            after=[],
            completed=True,
        ),
    )
    result = term_apply.invoke(
        CapabilityRequest(
            capability_id="lean.term.apply",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "statement": "P → P",
                "proof_prefix": ["intro P"],
                "term": "P",
            },
        )
    )
    from jacobian.contracts.lean_term_apply import LeanTermApplyOutput

    validated = LeanTermApplyOutput.model_validate(result.output)
    assert validated.term_application == "LEAN_EXACT_ELABORATION"
    assert validated.term_apply_uri == validated.transition_uri


# ---------------------------------------------------------------------------
# regression: metavariable fields rejects goal-count mismatch (comment J)
# ---------------------------------------------------------------------------


def test_metavariable_fields_rejects_goal_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_state, _, _, metavariable, _ = _adapters(tmp_path)
    _stub_apply_runtime(
        monkeypatch,
        proof_state,
        lambda: _responses(
            before=["P Q : Prop\n⊢ P", "P Q : Prop\n⊢ Q"],
            after=["P Q : Prop\n⊢ P", "P Q : Prop\n⊢ Q"],
        ),
    )
    opened = proof_state.invoke(
        CapabilityRequest(
            capability_id="lean.proof_state.apply_tactic",
            mode=CapabilityMode.EXPLORE,
            input={
                "environment": "CORE",
                "statement": "(P Q : Prop) → P ∧ Q",
                "proof_prefix": ["intro P Q"],
                "tactic": "constructor",
            },
        )
    )
    state_uri = opened.output["successor_states"][0]["state_uri"]
    structured, elaboration = _metavariable_fixture()
    # Return only one metavariable for a two-goal state.
    _stub_metavariable_runtime(
        monkeypatch,
        metavariable,
        structured=structured,
        elaboration=elaboration,
        before=["P Q : Prop\n⊢ P", "P Q : Prop\n⊢ Q"],
    )
    with pytest.raises(CapabilityInvocationError) as raised:
        metavariable.invoke(
            CapabilityRequest(
                capability_id="lean.proof_state.metavariable_fields",
                mode=CapabilityMode.EXPLORE,
                input={"environment": "CORE", "state_uri": state_uri},
            )
        )
    assert raised.value.diagnostic.code == "LEAN_METAVARIABLE_FIELDS_EXTRACTION_FAILED"


# ---------------------------------------------------------------------------
# regression: term length bound accounts for "exact " prefix (comment K)
# ---------------------------------------------------------------------------


def test_term_apply_rejects_term_exceeding_delegated_tactic_bound() -> None:
    from jacobian.contracts.lean_term_apply import LeanTermApplyRequest

    # A term of 995 chars + "exact " (6) = 1001, exceeding the 1000-char
    # tactic bound on LeanProofStateRequest. The term bound must reject this.
    with pytest.raises(ValueError):
        LeanTermApplyRequest.model_validate(
            {
                "environment": "CORE",
                "statement": "True",
                "term": "x" * 995,
            }
        )


def test_term_apply_accepts_term_at_corrected_bound() -> None:
    from jacobian.contracts.lean_term_apply import LeanTermApplyRequest

    # A term of 994 chars + "exact " (6) = 1000, exactly at the tactic bound.
    request = LeanTermApplyRequest.model_validate(
        {
            "environment": "CORE",
            "statement": "True",
            "term": "x" * 994,
        }
    )
    assert len(request.term) == 994


# ---------------------------------------------------------------------------
# regression: descriptors advertise accepted artifact types and read_only
# (comments G + L)
# ---------------------------------------------------------------------------


def test_inspect_descriptor_advertises_state_schema_and_read_only(
    tmp_path: Path,
) -> None:
    _, _, inspect, _, resources = _adapters(tmp_path)
    desc = inspect.descriptor
    from jacobian.contracts.capabilities import CapabilityInputKind

    assert desc.read_only is True
    assert CapabilityInputKind.TYPED_ARTIFACT in desc.accepted_input_kinds
    assert resources.state_schema_uri in desc.accepted_artifact_types


def test_metavariable_descriptor_advertises_state_schema(tmp_path: Path) -> None:
    _, _, _, metavariable, resources = _adapters(tmp_path)
    desc = metavariable.descriptor
    from jacobian.contracts.capabilities import CapabilityInputKind

    assert CapabilityInputKind.TYPED_ARTIFACT in desc.accepted_input_kinds
    assert resources.state_schema_uri in desc.accepted_artifact_types


def test_term_apply_descriptor_advertises_state_schema(tmp_path: Path) -> None:
    _, term_apply, _, _, resources = _adapters(tmp_path)
    desc = term_apply.descriptor
    from jacobian.contracts.capabilities import CapabilityInputKind

    assert CapabilityInputKind.TYPED_ARTIFACT in desc.accepted_input_kinds
    assert resources.state_schema_uri in desc.accepted_artifact_types
