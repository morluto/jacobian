from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian.eval.trajectory_state import (
    BindingValidity,
    CandidateState,
    CleanRoomTerminalEvidence,
    MilestoneKind,
    ReasoningProtocolState,
    StateBoundary,
    TerminalAcceptance,
    TrajectoryExtraction,
    TrajectoryStateError,
    extract_codex_trajectory,
)

RUN_ID = "00000000-0000-4000-8000-000000000000"
CALL_IDS = tuple(f"{index:08d}-0000-4000-8000-000000000000" for index in range(1, 20))
ARTIFACT_A = "artifact://sha256/" + "a" * 64
ARTIFACT_B = "artifact://sha256/" + "b" * 64
RECORD = "artifact://sha256/" + "c" * 64
ROOT = Path(__file__).resolve().parents[3]


def _event(
    tool: str,
    arguments: dict[str, object],
    response: dict[str, object] | None,
) -> dict[str, object]:
    result: dict[str, object] = {"isError": response is None}
    if response is not None:
        result.update(
            {
                "structured_content": response,
                "content": [{"type": "text", "text": json.dumps(response)}],
            }
        )
    return {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "server": "jacobian",
            "tool": tool,
            "arguments": arguments,
            "status": "completed",
            "result": result,
        },
    }


def _reasoning(
    phase: str,
    summary: str,
    *,
    call_id: str | None = None,
    capability_id: str = "polynomial.map.inverse.candidate_synthesize",
    mode: str = "EXPLORE",
) -> dict[str, object]:
    arguments: dict[str, object] = {"phase": phase, "summary": summary}
    response: dict[str, object] = {"run_id": RUN_ID}
    if phase != "PLAN":
        arguments["run_id"] = RUN_ID
    if phase == "BEFORE_TOOL":
        arguments.update({"capability_id": capability_id, "mode": mode})
        response["call_id"] = call_id
    elif phase == "AFTER_TOOL":
        arguments.update(
            {
                "call_id": call_id,
                "interpretation_status": "INTERPRETED",
                "reported_execution_status": "COMPLETED",
                "reported_assurance_level": "COMPUTED",
                "reported_completeness_status": "COMPLETE",
            }
        )
    elif phase == "FINAL":
        response["state"] = "FINALIZED"
    return _event("reasoning.write", arguments, response)


def _math(
    call_id: str,
    capability_id: str,
    response: dict[str, object] | None,
) -> dict[str, object]:
    mode = response.get("mode", "EXPLORE") if response is not None else "EXPLORE"
    return _event(
        "math.run",
        {
            "capability_id": capability_id,
            "mode": mode,
            "payload": {},
            "reasoning_run_id": RUN_ID,
            "reasoning_call_id": call_id,
        },
        response,
    )


def _result(
    capability_id: str,
    *,
    output: dict[str, object] | None = None,
    execution: str = "COMPLETED",
    assurance: str = "COMPUTED",
    verification_record_uri: str | None = None,
    completeness: str = "NOT_APPLICABLE",
    artifacts: list[str] | None = None,
    obligations: list[dict[str, object]] | None = None,
    scope: dict[str, object] | None = None,
    diagnostics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    mode = "VERIFY" if assurance == "VERIFIED" else "EXPLORE"
    completeness_assurance = (
        "VERIFIED"
        if assurance == "VERIFIED" and completeness == "COMPLETE"
        else "HEURISTIC"
    )
    return {
        "response_version": "2",
        "capability_id": capability_id,
        "capability_version": "fixture-v1",
        "mode": mode,
        "execution": {"status": execution},
        "output": output or {},
        "scope": scope,
        "completeness": {
            "status": completeness,
            "basis": "fixture completeness declaration",
            "assurance_level": completeness_assurance,
            "verification_record_uri": (
                verification_record_uri
                if completeness_assurance == "VERIFIED"
                else None
            ),
        },
        "relationships": [],
        "obligations": obligations or [],
        "diagnostics": diagnostics or [],
        "assurance": {
            "level": assurance,
            "basis": "fixture assurance declaration",
            "verification_record_uri": verification_record_uri,
        },
        "artifact_uris": artifacts or [],
    }


def _write(tmp_path: Path, events: list[dict[str, object]]) -> Path:
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def _cycle(
    call_id: str,
    capability_id: str,
    response: dict[str, object] | None,
    *,
    after: str,
) -> list[dict[str, object]]:
    mode_value = response.get("mode", "EXPLORE") if response is not None else "EXPLORE"
    mode = mode_value if isinstance(mode_value, str) else "EXPLORE"
    return [
        _reasoning(
            "BEFORE_TOOL",
            "Invoke the bounded operation.",
            call_id=call_id,
            capability_id=capability_id,
            mode=mode,
        ),
        _math(call_id, capability_id, response),
        _reasoning("AFTER_TOOL", after, call_id=call_id),
    ]


def test_extracts_typed_state_without_treating_boundaries_as_progress(
    tmp_path: Path,
) -> None:
    capability_id = "polynomial.map.inverse.candidate_synthesize"
    candidate: dict[str, object] = {
        "candidate_inverse_map": {"variables": ["x", "y"], "components": []}
    }
    response = _result(
        capability_id,
        output=candidate,
        artifacts=[ARTIFACT_A],
        scope={"parameters": {"domain": "QQ"}},
        completeness="COMPLETE",
    )
    events = [_reasoning("PLAN", "Find and check a polynomial inverse.")]
    events.extend(
        _cycle(
            CALL_IDS[0],
            capability_id,
            response,
            after="A candidate was produced; it still needs checking.",
        )
    )
    events.append(_reasoning("FINAL", "Report only the checked conclusion."))

    extraction = extract_codex_trajectory(
        _write(tmp_path, events), task_family="polynomial-map-inverse"
    )

    assert [state.boundary for state in extraction.states] == [
        StateBoundary.PLAN,
        StateBoundary.TOOL_RESULT,
        StateBoundary.AFTER_TOOL,
        StateBoundary.FINAL,
    ]
    assert extraction.states[0].milestone_eligible is False
    tool_state = extraction.states[1]
    assert set(tool_state.milestone_kinds) >= {
        MilestoneKind.OBJECT_ADDED,
        MilestoneKind.ARTIFACT_ADDED,
        MilestoneKind.CANDIDATE_PRODUCED,
        MilestoneKind.SCOPE_CHANGED,
        MilestoneKind.COMPLETENESS_CHANGED,
    }
    assert tool_state.hard_state.candidate_state is CandidateState.PRESENT
    assert tool_state.hard_state.binding_validity is BindingValidity.UNKNOWN
    assert extraction.states[2].milestone_eligible is False
    assert extraction.states[2].soft_state is not None
    assert (
        extraction.states[2].soft_state.latest_after_tool_summary
        == "A candidate was produced; it still needs checking."
    )
    assert extraction.states[-1].milestone_eligible is False
    assert (
        extraction.states[-1].hard_state.reasoning_protocol_state
        is ReasoningProtocolState.FINALIZED
    )
    assert extraction.assurance_authority is False


def test_volatile_output_metadata_does_not_create_a_new_typed_object(
    tmp_path: Path,
) -> None:
    capability_id = "integer.compute.gcd"
    base_output = {"result": {"value": "6"}, "backend_version": "sympy-v1"}
    events = [
        _reasoning("PLAN", "Compute the same result twice with volatile metadata.")
    ]
    events.extend(
        _cycle(
            CALL_IDS[0],
            capability_id,
            _result(capability_id, output=base_output),
            after="The first computation produced a typed object.",
        )
    )
    warmer_output = {
        "result": {"value": "6"},
        "backend_version": "sympy-v1",
        "cache_hit": True,
    }
    events.extend(
        _cycle(
            CALL_IDS[1],
            capability_id,
            _result(capability_id, output=warmer_output),
            after="A cache hit changes only volatile metadata, not identity.",
        )
    )
    tool_states = [
        state
        for state in extract_codex_trajectory(
            _write(tmp_path, events), task_family="exact-arithmetic"
        ).states
        if state.boundary is StateBoundary.TOOL_RESULT
    ]

    assert tool_states[0].milestone_eligible is True
    assert MilestoneKind.OBJECT_ADDED in tool_states[0].milestone_kinds
    assert tool_states[1].milestone_eligible is False
    assert MilestoneKind.OBJECT_ADDED not in tool_states[1].milestone_kinds


def test_repeated_call_and_rewritten_prose_cannot_create_fake_progress(
    tmp_path: Path,
) -> None:
    capability_id = "polynomial.map.inverse.candidate_synthesize"
    response = _result(
        capability_id,
        output={"candidate_inverse_map": {"components": ["x", "y"]}},
        artifacts=[ARTIFACT_A],
        scope={"parameters": {"domain": "QQ"}},
    )
    events = [_reasoning("PLAN", "First wording.")]
    events.extend(
        _cycle(CALL_IDS[0], capability_id, response, after="First interpretation.")
    )
    events.extend(
        _cycle(
            CALL_IDS[1],
            capability_id,
            response,
            after="Longer rewritten prose claiming much more progress.",
        )
    )
    events.extend(
        _cycle(
            CALL_IDS[2],
            "integer.compute.gcd",
            _result("integer.compute.gcd"),
            after="A completed call with no mathematical object is not progress.",
        )
    )
    events.append(_reasoning("FINAL", "A verbose final rewrite."))

    states = extract_codex_trajectory(
        _write(tmp_path, events), task_family="polynomial-map-inverse"
    ).states
    tool_states = [
        state for state in states if state.boundary is StateBoundary.TOOL_RESULT
    ]

    assert tool_states[0].milestone_eligible is True
    assert tool_states[1].milestone_eligible is False
    assert tool_states[1].milestone_kinds == ()
    assert all(
        not state.milestone_eligible
        for state in states
        if state.boundary
        in {StateBoundary.PLAN, StateBoundary.AFTER_TOOL, StateBoundary.FINAL}
    )


def test_tool_call_alone_timeout_and_incomplete_result_have_zero_milestones(
    tmp_path: Path,
) -> None:
    events = [_reasoning("PLAN", "Inspect a bounded search honestly.")]
    events.extend(
        _cycle(
            CALL_IDS[0],
            "polynomial.map.collision.search",
            _result(
                "polynomial.map.collision.search",
                output={"candidate_witness": {"point": [1, 2]}},
                execution="TIMEOUT",
                completeness="UNKNOWN",
            ),
            after="The timeout is not a conclusion.",
        )
    )
    events.extend(
        _cycle(
            CALL_IDS[1],
            "integer.compute.gcd",
            None,
            after="No result was available.",
        )
    )

    states = extract_codex_trajectory(
        _write(tmp_path, events), task_family="bounded-search"
    ).states
    tool_states = [
        state for state in states if state.boundary is StateBoundary.TOOL_RESULT
    ]

    assert all(state.milestone_eligible is False for state in tool_states)
    assert all(state.hard_state.typed_objects == () for state in tool_states)
    assert tool_states[0].hard_state.candidate_state is CandidateState.ABSENT


def test_malformed_or_substituted_typed_results_cannot_create_objects(
    tmp_path: Path,
) -> None:
    requested = "integer.compute.gcd"
    events = [_reasoning("PLAN", "Reject malformed and substituted results.")]
    malformed = _result(requested, output={"result": {"value": "6"}})
    malformed["assurance"] = {"level": "VERIFIED"}
    events.extend(
        _cycle(
            CALL_IDS[0],
            requested,
            malformed,
            after="The malformed assurance must not become progress.",
        )
    )
    events.extend(
        _cycle(
            CALL_IDS[1],
            requested,
            _result(
                "integer.compute.lcm",
                output={"result": {"value": "420"}},
            ),
            after="A substituted capability result must not become progress.",
        )
    )

    tool_states = [
        state
        for state in extract_codex_trajectory(
            _write(tmp_path, events), task_family="exact-arithmetic"
        ).states
        if state.boundary is StateBoundary.TOOL_RESULT
    ]

    assert tool_states[0].milestone_eligible is False
    assert tool_states[0].hard_state.typed_objects == ()
    assert tool_states[1].milestone_kinds == (MilestoneKind.BINDING_BECAME_INVALID,)
    assert tool_states[1].hard_state.typed_objects == ()


def test_rejection_repair_obligations_and_verified_binding_are_milestones(
    tmp_path: Path,
) -> None:
    producer = "polynomial.map.inverse.candidate_synthesize"
    checker = "polynomial.map.inverse.verify"
    obligation = ARTIFACT_B
    events = [_reasoning("PLAN", "Produce, reject, repair, and verify.")]
    events.extend(
        _cycle(
            CALL_IDS[0],
            producer,
            _result(
                producer,
                output={"candidate_inverse_map": {"components": ["x", "0"]}},
                obligations=[{"obligation_uri": obligation, "status": "OPEN"}],
            ),
            after="The first candidate has an open obligation.",
        )
    )
    events.extend(
        _cycle(
            CALL_IDS[1],
            checker,
            _result(checker, output={"status": "REJECTED"}),
            after="The checker rejected the first candidate.",
        )
    )
    events.extend(
        _cycle(
            CALL_IDS[2],
            producer,
            _result(
                producer,
                output={"candidate_inverse_map": {"components": ["x", "y"]}},
            ),
            after="A structurally different candidate repairs the rejection.",
        )
    )
    events.extend(
        _cycle(
            CALL_IDS[3],
            checker,
            _result(
                checker,
                output={"status": "VERIFIED"},
                assurance="VERIFIED",
                verification_record_uri=RECORD,
                artifacts=[RECORD],
                obligations=[
                    {
                        "obligation_uri": obligation,
                        "status": "DISCHARGED",
                        "verification_record_uri": RECORD,
                    }
                ],
            ),
            after="The independent checker accepted the repaired candidate.",
        )
    )

    tool_states = [
        state
        for state in extract_codex_trajectory(
            _write(tmp_path, events), task_family="polynomial-map-inverse"
        ).states
        if state.boundary is StateBoundary.TOOL_RESULT
    ]

    assert MilestoneKind.OBLIGATION_OPENED in tool_states[0].milestone_kinds
    assert MilestoneKind.CHECKER_REJECTED in tool_states[1].milestone_kinds
    assert tool_states[1].hard_state.candidate_state is CandidateState.REJECTED
    assert MilestoneKind.CANDIDATE_REPAIRED in tool_states[2].milestone_kinds
    assert set(tool_states[3].milestone_kinds) >= {
        MilestoneKind.CHECKER_ACCEPTED,
        MilestoneKind.OBLIGATION_DISCHARGED,
        MilestoneKind.BINDING_BECAME_VALID,
    }
    assert tool_states[3].hard_state.candidate_state is CandidateState.VERIFIED
    assert tool_states[3].hard_state.binding_validity is BindingValidity.VALID


def test_scope_and_binding_diagnostics_are_structural_not_prose(tmp_path: Path) -> None:
    events = [_reasoning("PLAN", "Respect the exact bounded scope.")]
    events.extend(
        _cycle(
            CALL_IDS[0],
            "polynomial.map.collision.search",
            _result(
                "polynomial.map.collision.search",
                execution="ERROR",
                completeness="UNKNOWN",
                diagnostics=[
                    {
                        "code": "SCOPE_ESCALATION_FORBIDDEN",
                        "stage": "scope_validation",
                        "message": "The requested scope exceeds the declared bound.",
                    },
                    {
                        "code": "STALE_ARTIFACT_BINDING",
                        "stage": "artifact_binding",
                        "message": "The supplied artifact binding is stale.",
                    },
                ],
            ),
            after="I claim this was successful despite the diagnostics.",
        )
    )

    tool_state = next(
        state
        for state in extract_codex_trajectory(
            _write(tmp_path, events), task_family="bounded-collision-search"
        ).states
        if state.boundary is StateBoundary.TOOL_RESULT
    )

    assert set(tool_state.milestone_kinds) == {
        MilestoneKind.BINDING_BECAME_INVALID,
        MilestoneKind.SCOPE_ESCALATION_REJECTED,
    }
    assert tool_state.hard_state.binding_validity is BindingValidity.INVALID
    assert tool_state.hard_state.candidate_state is CandidateState.ABSENT


def test_terminal_evidence_is_clean_room_fail_closed_and_not_assurance(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, [_reasoning("PLAN", "Solve exactly.")])
    source_digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = CleanRoomTerminalEvidence(
        verifier_digest="sha256:" + "d" * 64,
        clean_room=True,
        verifier_execution_status="COMPLETED",
        acceptance=TerminalAcceptance.ACCEPTED,
        input_binding_valid=True,
        artifact_binding_valid=True,
        source_binding_digest=source_digest,
    )

    extraction = extract_codex_trajectory(
        path, task_family="exact-arithmetic", terminal_evidence=evidence
    )

    terminal = extraction.states[-1]
    assert terminal.boundary is StateBoundary.TERMINAL
    assert terminal.milestone_eligible is False
    assert terminal.hard_state.assurance_level is None
    assert terminal.hard_state.binding_validity is BindingValidity.VALID
    assert extraction.assurance_authority is False
    with pytest.raises(ValidationError, match="inconclusive"):
        CleanRoomTerminalEvidence(
            verifier_digest="sha256:" + "e" * 64,
            clean_room=True,
            verifier_execution_status="TIMEOUT",
            acceptance=TerminalAcceptance.ACCEPTED,
            source_binding_digest=source_digest,
        )
    with pytest.raises(ValidationError, match="invalid bindings"):
        CleanRoomTerminalEvidence(
            verifier_digest="sha256:" + "e" * 64,
            clean_room=True,
            verifier_execution_status="COMPLETED",
            acceptance=TerminalAcceptance.ACCEPTED,
            input_binding_valid=False,
            source_binding_digest=source_digest,
        )


def test_strict_source_and_closed_models_reject_malformed_input(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"type":"item.completed"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(TrajectoryStateError, match="line 2"):
        extract_codex_trajectory(malformed, task_family="test")

    extraction = TrajectoryExtraction(
        source_digest="sha256:" + "f" * 64,
        task_family="test",
        states=(),
    )
    payload = extraction.model_dump(mode="json")
    payload["invented"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        TrajectoryExtraction.model_validate(payload)


def test_committed_json_schema_matches_typed_contract(tmp_path: Path) -> None:
    path = _write(tmp_path, [_reasoning("PLAN", "Inspect the state contract.")])
    extraction = extract_codex_trajectory(path, task_family="schema-test")
    schema = json.loads(
        (
            ROOT / "docs/reference/evaluations/schemas/trajectory-state-v1.schema.json"
        ).read_text(encoding="utf-8")
    )

    assert schema == TrajectoryExtraction.model_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(extraction.model_dump(mode="json"))


def test_foreign_mcp_server_tool_calls_are_not_trajectory_state(
    tmp_path: Path,
) -> None:
    events = [_reasoning("PLAN", "Use a foreign server masquerading as Jacobian.")]
    events.extend(
        _cycle(
            CALL_IDS[0],
            "integer.compute.gcd",
            _result(
                "integer.compute.gcd",
                output={"result": {"value": "6"}},
            ),
            after="A foreign server response must not become state.",
        )
    )
    for event in events:
        item = event.get("item", {})
        if isinstance(item, dict) and item.get("tool") == "math.run":
            item["server"] = "foreign-mcp-server"
    path = _write(tmp_path, events)
    extraction = extract_codex_trajectory(path, task_family="foreign-server-test")

    assert extraction.states[0].boundary is StateBoundary.PLAN
    assert all(
        state.boundary is not StateBoundary.TOOL_RESULT for state in extraction.states
    )


def test_domain_specific_verified_status_recognizes_via_assurance_contract(
    tmp_path: Path,
) -> None:
    checker = "matrix.determinant.verify"
    events = [_reasoning("PLAN", "Verify a determinant with a domain-specific status.")]
    events.extend(
        _cycle(
            CALL_IDS[0],
            "matrix.determinant.compute",
            _result(
                "matrix.determinant.compute",
                output={
                    "determinant": "42",
                    "certificate_available": False,
                },
            ),
            after="A determinant was computed without a certificate.",
        )
    )
    events.extend(
        _cycle(
            CALL_IDS[1],
            checker,
            _result(
                checker,
                output={"status": "VERIFIED_DETERMINANT"},
                assurance="VERIFIED",
                verification_record_uri=RECORD,
                artifacts=[RECORD],
            ),
            after="The independent verifier accepted the determinant.",
        )
    )
    tool_states = [
        state
        for state in extract_codex_trajectory(
            _write(tmp_path, events), task_family="matrix-determinant"
        ).states
        if state.boundary is StateBoundary.TOOL_RESULT
    ]
    assert MilestoneKind.CHECKER_ACCEPTED in tool_states[1].milestone_kinds
    assert tool_states[1].hard_state.candidate_state is CandidateState.VERIFIED
    assert tool_states[1].hard_state.binding_validity is BindingValidity.VALID


def test_committed_real_codex_sample_replays_and_binds_every_file() -> None:
    fixture = Path(__file__).parent / "fixtures/trajectory_state/pr1_gcd_real_codex"
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        actual = "sha256:" + hashlib.sha256((fixture / name).read_bytes()).hexdigest()
        assert actual == expected
    prompt_digest = (
        "sha256:"
        + hashlib.sha256(
            (fixture / manifest["prompt"]["path"]).read_bytes()
        ).hexdigest()
    )
    assert prompt_digest == manifest["prompt"]["digest"]

    extraction = extract_codex_trajectory(
        fixture / "codex.jsonl",
        task_family=manifest["extractor"]["task_family"],
    )
    committed = json.loads((fixture / "extracted.json").read_text(encoding="utf-8"))

    assert extraction.model_dump(mode="json") == committed
    assert sum(state.milestone_eligible for state in extraction.states) == 1
    assert extraction.states[0].boundary is StateBoundary.PLAN
    assert extraction.states[-1].boundary is StateBoundary.FINAL
