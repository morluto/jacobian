"""Deterministic observable trajectory-state extraction for evaluations.

The extractor consumes Codex JSONL and records only model-authored external
reasoning summaries plus typed Jacobian results.  It is deliberately outside
the mathematical assurance path: no state, milestone, or terminal label can
authorize ``VERIFIED``.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, ValidationError, model_validator

from jacobian.canonical import CanonicalizationError, canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityResult,
)
from jacobian.contracts.results import ContractModel

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ARTIFACT_PATTERN = r"^artifact://sha256/[0-9a-f]{64}$"
_JACOBIAN_SERVER = "jacobian"
_CONTROL_OUTPUT_FIELDS = frozenset(
    {
        "accepted",
        "conclusion",
        "detail",
        "diagnostics",
        "execution_status",
        "found",
        "message",
        "status",
        "stop_reason",
        "success",
        "valid",
        "verified",
    }
)
_CANDIDATE_FIELD_PARTS = frozenset(
    {"candidate", "certificate", "counterexample", "witness"}
)
_CHECKER_ID_PARTS = frozenset({"audit", "check", "checker", "verify"})
_ACCEPTED_CHECKER_STATUSES = frozenset(
    {"ACCEPTED", "HOLDS", "PASS", "PASSED", "VALID", "VERIFIED"}
)
_REJECTED_CHECKER_STATUSES = frozenset(
    {"FAIL", "FAILED", "INVALID", "NOT_VERIFIED", "REJECTED"}
)
_NONCONCLUSIVE_EXECUTION = frozenset({"CANCELLED", "ERROR", "TIMEOUT"})
ArtifactRole = Literal[
    "MATHEMATICAL_RESULT",
    "OBLIGATION",
    "SCOPE",
    "VERIFICATION_RECORD",
]


class TrajectoryStateError(ValueError):
    """A trajectory source cannot support deterministic state extraction."""


class StateBoundary(StrEnum):
    PLAN = "PLAN"
    TOOL_RESULT = "TOOL_RESULT"
    AFTER_TOOL = "AFTER_TOOL"
    FINAL = "FINAL"
    TERMINAL = "TERMINAL"


class ReasoningProtocolState(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    READY_TO_INVOKE = "READY_TO_INVOKE"
    AWAITING_AFTER_TOOL = "AWAITING_AFTER_TOOL"
    FINALIZED = "FINALIZED"
    INVALID = "INVALID"


class CandidateState(StrEnum):
    ABSENT = "ABSENT"
    PRESENT = "PRESENT"
    REJECTED = "REJECTED"
    CHECKED = "CHECKED"
    VERIFIED = "VERIFIED"


class CheckerState(StrEnum):
    NOT_CHECKED = "NOT_CHECKED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class BindingValidity(StrEnum):
    UNKNOWN = "UNKNOWN"
    VALID = "VALID"
    INVALID = "INVALID"


class TerminalAcceptance(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class MilestoneKind(StrEnum):
    OBJECT_ADDED = "OBJECT_ADDED"
    ARTIFACT_ADDED = "ARTIFACT_ADDED"
    CANDIDATE_PRODUCED = "CANDIDATE_PRODUCED"
    CANDIDATE_REPAIRED = "CANDIDATE_REPAIRED"
    CHECKER_ACCEPTED = "CHECKER_ACCEPTED"
    CHECKER_REJECTED = "CHECKER_REJECTED"
    OBLIGATION_OPENED = "OBLIGATION_OPENED"
    OBLIGATION_DISCHARGED = "OBLIGATION_DISCHARGED"
    BINDING_BECAME_VALID = "BINDING_BECAME_VALID"
    BINDING_BECAME_INVALID = "BINDING_BECAME_INVALID"
    SCOPE_CHANGED = "SCOPE_CHANGED"
    SCOPE_ESCALATION_REJECTED = "SCOPE_ESCALATION_REJECTED"
    COMPLETENESS_CHANGED = "COMPLETENESS_CHANGED"
    ASSURANCE_CHANGED = "ASSURANCE_CHANGED"


class TypedObjectRef(ContractModel):
    object_type: str = Field(min_length=1, max_length=256)
    content_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_capability_id: str = Field(min_length=3, max_length=128)


class ArtifactStateRef(ContractModel):
    artifact_uri: str = Field(pattern=_ARTIFACT_PATTERN)
    role: ArtifactRole
    source_capability_id: str = Field(min_length=3, max_length=128)


class TrajectoryHardState(ContractModel):
    state_schema_version: Literal["1"] = "1"
    task_family: str = Field(min_length=1, max_length=128)
    typed_objects: tuple[TypedObjectRef, ...] = ()
    artifacts: tuple[ArtifactStateRef, ...] = ()
    candidate_state: CandidateState = CandidateState.ABSENT
    latest_candidate_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    checker_state: CheckerState = CheckerState.NOT_CHECKED
    open_obligation_uris: tuple[str, ...] = ()
    discharged_obligation_uris: tuple[str, ...] = ()
    execution_status: str | None = Field(default=None, min_length=1, max_length=64)
    completeness_status: str | None = Field(default=None, min_length=1, max_length=64)
    completeness_assurance: str | None = Field(
        default=None, min_length=1, max_length=64
    )
    assurance_level: str | None = Field(default=None, min_length=1, max_length=64)
    scope_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    scope_escalation_errors: tuple[str, ...] = ()
    binding_validity: BindingValidity = BindingValidity.UNKNOWN
    latest_meaningful_transitions: tuple[MilestoneKind, ...] = ()
    reasoning_protocol_state: ReasoningProtocolState = (
        ReasoningProtocolState.NOT_STARTED
    )

    @model_validator(mode="after")
    def require_consistent_sets_and_candidate(self) -> Self:
        if set(self.open_obligation_uris) & set(self.discharged_obligation_uris):
            raise ValueError("an obligation cannot be both open and discharged")
        if len(set(self.open_obligation_uris)) != len(self.open_obligation_uris):
            raise ValueError("open obligations must be unique")
        if len(set(self.discharged_obligation_uris)) != len(
            self.discharged_obligation_uris
        ):
            raise ValueError("discharged obligations must be unique")
        if self.candidate_state is CandidateState.ABSENT:
            if self.latest_candidate_digest is not None:
                raise ValueError("an absent candidate cannot have a digest")
        elif self.latest_candidate_digest is None:
            raise ValueError("a present or checked candidate requires a digest")
        return self


class TrajectorySoftState(ContractModel):
    """Optional model-authored external summaries; never hidden chain-of-thought."""

    soft_state_schema_version: Literal["1"] = "1"
    plan_summary: str | None = Field(default=None, max_length=512)
    latest_after_tool_summary: str | None = Field(default=None, max_length=512)
    final_summary: str | None = Field(default=None, max_length=512)


class ExtractedTrajectoryState(ContractModel):
    index: int = Field(ge=0, strict=True)
    source_event_index: int = Field(ge=0, strict=True)
    boundary: StateBoundary
    hard_state: TrajectoryHardState
    soft_state: TrajectorySoftState | None = None
    hard_state_digest: str = Field(pattern=_DIGEST_PATTERN)
    changed_fields: tuple[str, ...] = ()
    milestone_kinds: tuple[MilestoneKind, ...] = ()
    milestone_eligible: bool
    milestone_reason: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def bind_eligibility_to_kinds(self) -> Self:
        if self.milestone_eligible != bool(self.milestone_kinds):
            raise ValueError("milestone eligibility must equal presence of kinds")
        if self.milestone_kinds != self.hard_state.latest_meaningful_transitions:
            raise ValueError("milestone kinds must match the hard-state transition")
        return self


class CleanRoomTerminalEvidence(ContractModel):
    evidence_schema_version: Literal["1"] = "1"
    verifier_digest: str = Field(pattern=_DIGEST_PATTERN)
    clean_room: Literal[True]
    verifier_execution_status: Literal["COMPLETED", "TIMEOUT", "CANCELLED", "ERROR"]
    acceptance: TerminalAcceptance
    input_binding_valid: bool | None = None
    artifact_binding_valid: bool | None = None
    source_binding_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def fail_closed_on_noncompletion(self) -> Self:
        if (
            self.verifier_execution_status != "COMPLETED"
            and self.acceptance is not TerminalAcceptance.INCONCLUSIVE
        ):
            raise ValueError("non-completed verifier evidence is inconclusive")
        if self.acceptance is not TerminalAcceptance.INCONCLUSIVE and False in (
            self.input_binding_valid,
            self.artifact_binding_valid,
        ):
            raise ValueError(
                "conclusive terminal evidence cannot have invalid bindings"
            )
        return self


class TrajectoryExtraction(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-state-v1.schema.json"
        },
    )

    extraction_schema_version: Literal["1"] = "1"
    extractor_id: Literal["jacobian.observable-trajectory-state.v1"] = (
        "jacobian.observable-trajectory-state.v1"
    )
    source_format: Literal["codex-jsonl"] = "codex-jsonl"
    source_digest: str = Field(pattern=_DIGEST_PATTERN)
    task_family: str = Field(min_length=1, max_length=128)
    states: tuple[ExtractedTrajectoryState, ...]
    terminal_evidence: CleanRoomTerminalEvidence | None = None
    assurance_authority: Literal[False] = False

    @model_validator(mode="after")
    def require_ordered_states(self) -> Self:
        if tuple(state.index for state in self.states) != tuple(
            range(len(self.states))
        ):
            raise ValueError("state indices must be contiguous")
        return self


def _digest(value: object) -> str:
    try:
        encoded = canonicalize_json(value)
    except CanonicalizationError as exc:
        raise TrajectoryStateError(
            "trajectory value is not canonical exact JSON"
        ) from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _tool_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("__", ".").replace("_", ".").lower()
    for expected in ("math.run", "reasoning.write"):
        if normalized.endswith(expected):
            return expected
    return None


def _response_payload(item: dict[str, Any]) -> dict[str, Any] | None:
    result = item.get("result")
    if not isinstance(result, dict):
        return None
    for key in ("structured_content", "structuredContent"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    content = result.get("content")
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict) or not isinstance(block.get("text"), str):
            continue
        try:
            value = json.loads(block["text"])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _diagnostic_codes(response: dict[str, Any]) -> tuple[str, ...]:
    diagnostics = response.get("diagnostics")
    if not isinstance(diagnostics, list):
        return ()
    return tuple(
        sorted(
            {
                item["code"]
                for item in diagnostics
                if isinstance(item, dict) and isinstance(item.get("code"), str)
            }
        )
    )


def _candidate_values(value: object, prefix: str = "output") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            key_parts = set(key.lower().replace("-", "_").split("_"))
            if key_parts & _CANDIDATE_FIELD_PARTS and item not in (None, {}, []):
                found.append((path, item))
            else:
                found.extend(_candidate_values(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_candidate_values(item, f"{prefix}[{index}]"))
    return found


def _is_checker(capability_id: str) -> bool:
    return bool(
        set(capability_id.lower().replace("-", ".").split(".")) & _CHECKER_ID_PARTS
    )


def _status(value: object) -> str | None:
    if isinstance(value, str):
        return value.upper()
    if value is True:
        return "ACCEPTED"
    if value is False:
        return "REJECTED"
    return None


def _checker_outcome(
    capability_id: str,
    validated: CapabilityResult | None,
) -> CheckerState | None:
    if not _is_checker(capability_id) or validated is None:
        return None
    assurance = validated.assurance
    if assurance.level is CapabilityAssuranceLevel.VERIFIED:
        return CheckerState.ACCEPTED
    output = validated.output
    candidates = (
        output.get("status"),
        output.get("verdict"),
        output.get("accepted"),
        output.get("valid"),
        output.get("verified"),
    )
    statuses = {_status(value) for value in candidates}
    if statuses & _REJECTED_CHECKER_STATUSES:
        return CheckerState.REJECTED
    return None


def _meaningful_output(output: object) -> bool:
    return isinstance(output, dict) and any(
        key not in _CONTROL_OUTPUT_FIELDS and value not in (None, {}, [])
        for key, value in output.items()
    )


def _changed_fields(
    previous: TrajectoryHardState | None, current: TrajectoryHardState
) -> tuple[str, ...]:
    if previous is None:
        return tuple(type(current).model_fields)
    before = previous.model_dump(mode="json")
    after = current.model_dump(mode="json")
    return tuple(sorted(key for key in after if before.get(key) != after.get(key)))


class _MutableState:
    def __init__(self, task_family: str) -> None:
        self.task_family = task_family
        self.objects: dict[tuple[str, str, str], TypedObjectRef] = {}
        self.artifacts: dict[str, ArtifactStateRef] = {}
        self.candidate_state = CandidateState.ABSENT
        self.candidate_digest: str | None = None
        self.checker_state = CheckerState.NOT_CHECKED
        self.open_obligations: set[str] = set()
        self.discharged_obligations: set[str] = set()
        self.execution_status: str | None = None
        self.completeness_status: str | None = None
        self.completeness_assurance: str | None = None
        self.assurance_level: str | None = None
        self.scope_digest: str | None = None
        self.scope_escalations: set[str] = set()
        self.binding_validity = BindingValidity.UNKNOWN
        self.protocol_state = ReasoningProtocolState.NOT_STARTED
        self.plan: str | None = None
        self.after: str | None = None
        self.final: str | None = None
        self.run_id: str | None = None
        self.pending_call_id: str | None = None
        self.latest_transitions: tuple[MilestoneKind, ...] = ()

    def hard(self) -> TrajectoryHardState:
        return TrajectoryHardState(
            task_family=self.task_family,
            typed_objects=tuple(
                sorted(
                    self.objects.values(),
                    key=lambda item: (
                        item.object_type,
                        item.content_digest,
                        item.source_capability_id,
                    ),
                )
            ),
            artifacts=tuple(
                sorted(self.artifacts.values(), key=lambda item: item.artifact_uri)
            ),
            candidate_state=self.candidate_state,
            latest_candidate_digest=self.candidate_digest,
            checker_state=self.checker_state,
            open_obligation_uris=tuple(sorted(self.open_obligations)),
            discharged_obligation_uris=tuple(sorted(self.discharged_obligations)),
            execution_status=self.execution_status,
            completeness_status=self.completeness_status,
            completeness_assurance=self.completeness_assurance,
            assurance_level=self.assurance_level,
            scope_digest=self.scope_digest,
            scope_escalation_errors=tuple(sorted(self.scope_escalations)),
            binding_validity=self.binding_validity,
            latest_meaningful_transitions=self.latest_transitions,
            reasoning_protocol_state=self.protocol_state,
        )

    def soft(self) -> TrajectorySoftState | None:
        if self.plan is None and self.after is None and self.final is None:
            return None
        return TrajectorySoftState(
            plan_summary=self.plan,
            latest_after_tool_summary=self.after,
            final_summary=self.final,
        )


def _record_plan(state: _MutableState, summary: str, response: dict[str, Any]) -> None:
    if state.protocol_state is not ReasoningProtocolState.NOT_STARTED:
        state.protocol_state = ReasoningProtocolState.INVALID
    else:
        state.protocol_state = ReasoningProtocolState.READY
        run_id = response.get("run_id")
        state.run_id = run_id if isinstance(run_id, str) else None
    state.plan = summary


def _record_before_tool(
    state: _MutableState,
    response: dict[str, Any],
) -> None:
    if state.protocol_state is not ReasoningProtocolState.READY:
        state.protocol_state = ReasoningProtocolState.INVALID
        return
    call_id = response.get("call_id")
    if not isinstance(call_id, str):
        state.protocol_state = ReasoningProtocolState.INVALID
        return
    state.pending_call_id = call_id
    state.protocol_state = ReasoningProtocolState.READY_TO_INVOKE


def _record_after_tool(
    state: _MutableState,
    arguments: dict[str, Any],
    summary: str,
) -> None:
    if (
        state.protocol_state is not ReasoningProtocolState.AWAITING_AFTER_TOOL
        or arguments.get("call_id") != state.pending_call_id
    ):
        state.protocol_state = ReasoningProtocolState.INVALID
    else:
        state.protocol_state = ReasoningProtocolState.READY
        state.pending_call_id = None
    state.after = summary


def _record_final(state: _MutableState, summary: str) -> None:
    if state.protocol_state is not ReasoningProtocolState.READY:
        state.protocol_state = ReasoningProtocolState.INVALID
    else:
        state.protocol_state = ReasoningProtocolState.FINALIZED
    state.final = summary


def _record_reasoning(
    state: _MutableState,
    arguments: dict[str, Any],
    response: dict[str, Any],
) -> StateBoundary | None:
    phase = arguments.get("phase")
    summary = arguments.get("summary")
    if not isinstance(phase, str) or not isinstance(summary, str):
        state.protocol_state = ReasoningProtocolState.INVALID
        return None
    if phase == "PLAN":
        _record_plan(state, summary, response)
        return StateBoundary.PLAN
    if arguments.get("run_id") != state.run_id or state.run_id is None:
        state.protocol_state = ReasoningProtocolState.INVALID
    if phase == "BEFORE_TOOL":
        _record_before_tool(state, response)
        return None
    if phase == "AFTER_TOOL":
        _record_after_tool(state, arguments, summary)
        return StateBoundary.AFTER_TOOL
    if phase == "FINAL":
        _record_final(state, summary)
        return StateBoundary.FINAL
    state.protocol_state = ReasoningProtocolState.INVALID
    return None


def _artifact_role(uri: str, response: dict[str, Any]) -> ArtifactRole:
    obligations = response.get("obligations")
    if isinstance(obligations, list) and any(
        isinstance(item, dict) and item.get("obligation_uri") == uri
        for item in obligations
    ):
        return "OBLIGATION"
    assurance = response.get("assurance")
    completeness = response.get("completeness")
    if (
        isinstance(assurance, dict) and assurance.get("verification_record_uri") == uri
    ) or (
        isinstance(completeness, dict)
        and completeness.get("verification_record_uri") == uri
    ):
        return "VERIFICATION_RECORD"
    scope = response.get("scope")
    if isinstance(scope, dict) and scope.get("artifact_uri") == uri:
        return "SCOPE"
    return "MATHEMATICAL_RESULT"


def _record_binding(
    state: _MutableState,
    arguments: dict[str, Any],
) -> None:
    """Track reasoning-call protocol binding without treating a call as progress."""

    if state.protocol_state is ReasoningProtocolState.READY_TO_INVOKE:
        if (
            arguments.get("reasoning_run_id") == state.run_id
            and arguments.get("reasoning_call_id") == state.pending_call_id
        ):
            state.protocol_state = ReasoningProtocolState.AWAITING_AFTER_TOOL
        else:
            state.protocol_state = ReasoningProtocolState.INVALID
    elif state.protocol_state is not ReasoningProtocolState.NOT_STARTED:
        state.protocol_state = ReasoningProtocolState.INVALID


def _record_diagnostics(
    state: _MutableState,
    response: dict[str, Any],
) -> set[MilestoneKind]:
    kinds: set[MilestoneKind] = set()
    codes = _diagnostic_codes(response)
    if (
        any(
            any(part in code for part in ("BIND", "STALE", "SUBSTITUT"))
            for code in codes
        )
        and state.binding_validity is not BindingValidity.INVALID
    ):
        state.binding_validity = BindingValidity.INVALID
        kinds.add(MilestoneKind.BINDING_BECAME_INVALID)
    new_scope_errors = {
        code
        for code in codes
        if "SCOPE" in code
        and any(part in code for part in ("ESCALAT", "EXCEED", "GLOBAL", "OUTSIDE"))
    } - state.scope_escalations
    if new_scope_errors:
        state.scope_escalations.update(new_scope_errors)
        kinds.add(MilestoneKind.SCOPE_ESCALATION_REJECTED)
    return kinds


def _record_objects(
    state: _MutableState,
    capability_id: str,
    output: object,
) -> set[MilestoneKind]:
    kinds: set[MilestoneKind] = set()
    if _meaningful_output(output):
        ref = TypedObjectRef(
            object_type=f"{capability_id}.output",
            content_digest=_digest(output),
            source_capability_id=capability_id,
        )
        key = (ref.object_type, ref.content_digest, ref.source_capability_id)
        if key not in state.objects:
            state.objects[key] = ref
            kinds.add(MilestoneKind.OBJECT_ADDED)
    candidates = _candidate_values(output)
    if not candidates:
        return kinds
    candidate_type, candidate = candidates[-1]
    candidate_digest = _digest(candidate)
    ref = TypedObjectRef(
        object_type=f"{capability_id}.{candidate_type}",
        content_digest=candidate_digest,
        source_capability_id=capability_id,
    )
    key = (ref.object_type, ref.content_digest, ref.source_capability_id)
    if key not in state.objects:
        state.objects[key] = ref
    if candidate_digest != state.candidate_digest:
        repaired = state.candidate_state is CandidateState.REJECTED
        state.candidate_digest = candidate_digest
        state.candidate_state = CandidateState.PRESENT
        state.checker_state = CheckerState.NOT_CHECKED
        kinds.add(
            MilestoneKind.CANDIDATE_REPAIRED
            if repaired
            else MilestoneKind.CANDIDATE_PRODUCED
        )
    return kinds


def _record_artifacts(
    state: _MutableState,
    capability_id: str,
    response: dict[str, Any],
) -> set[MilestoneKind]:
    kinds: set[MilestoneKind] = set()
    artifact_uris = response.get("artifact_uris")
    if not isinstance(artifact_uris, list):
        return kinds
    for uri in artifact_uris:
        if not isinstance(uri, str) or uri in state.artifacts:
            continue
        try:
            ref = ArtifactStateRef(
                artifact_uri=uri,
                role=_artifact_role(uri, response),
                source_capability_id=capability_id,
            )
        except ValueError:
            continue
        state.artifacts[uri] = ref
        kinds.add(MilestoneKind.ARTIFACT_ADDED)
    return kinds


def _record_obligations(
    state: _MutableState,
    response: dict[str, Any],
) -> set[MilestoneKind]:
    kinds: set[MilestoneKind] = set()
    obligations = response.get("obligations")
    if not isinstance(obligations, list):
        return kinds
    for item in obligations:
        if not isinstance(item, dict) or not isinstance(
            item.get("obligation_uri"), str
        ):
            continue
        uri = item["obligation_uri"]
        if item.get("status") == "DISCHARGED":
            if uri not in state.discharged_obligations:
                state.open_obligations.discard(uri)
                state.discharged_obligations.add(uri)
                kinds.add(MilestoneKind.OBLIGATION_DISCHARGED)
        elif uri not in state.open_obligations:
            state.open_obligations.add(uri)
            kinds.add(MilestoneKind.OBLIGATION_OPENED)
    return kinds


def _record_scope_completeness_assurance(
    state: _MutableState,
    response: dict[str, Any],
) -> set[MilestoneKind]:
    kinds: set[MilestoneKind] = set()
    scope = response.get("scope")
    if isinstance(scope, dict):
        scope_digest = _digest(scope)
        if scope_digest != state.scope_digest:
            state.scope_digest = scope_digest
            kinds.add(MilestoneKind.SCOPE_CHANGED)
    completeness = response.get("completeness")
    if isinstance(completeness, dict):
        completeness_status = completeness.get("status")
        completeness_assurance = completeness.get("assurance_level")
        previous_status = state.completeness_status
        previous_assurance = state.completeness_assurance
        changed = False
        if isinstance(completeness_status, str) and (
            completeness_status != state.completeness_status
        ):
            state.completeness_status = completeness_status
            changed = True
        if isinstance(completeness_assurance, str) and (
            completeness_assurance != state.completeness_assurance
        ):
            state.completeness_assurance = completeness_assurance
            changed = True
        if changed and (
            previous_status is not None
            or previous_assurance is not None
            or completeness_status in {"PARTIAL", "COMPLETE"}
        ):
            kinds.add(MilestoneKind.COMPLETENESS_CHANGED)
    assurance = response.get("assurance")
    if isinstance(assurance, dict) and isinstance(assurance.get("level"), str):
        level = assurance["level"]
        previous_level = state.assurance_level
        if level != state.assurance_level:
            state.assurance_level = level
            if previous_level is not None or level == "VERIFIED":
                kinds.add(MilestoneKind.ASSURANCE_CHANGED)
    return kinds


def _record_checker(
    state: _MutableState,
    capability_id: str,
    validated: CapabilityResult | None,
) -> set[MilestoneKind]:
    kinds: set[MilestoneKind] = set()
    checker = _checker_outcome(capability_id, validated)
    if checker is CheckerState.REJECTED and state.checker_state is not checker:
        state.checker_state = checker
        if state.candidate_digest is not None:
            state.candidate_state = CandidateState.REJECTED
        kinds.add(MilestoneKind.CHECKER_REJECTED)
    elif checker is CheckerState.ACCEPTED and state.checker_state is not checker:
        state.checker_state = checker
        if state.candidate_digest is not None:
            record_uri = None
            if validated is not None:
                record_uri = validated.assurance.verification_record_uri
            state.candidate_state = (
                CandidateState.VERIFIED
                if state.assurance_level == "VERIFIED" and isinstance(record_uri, str)
                else CandidateState.CHECKED
            )
            if (
                state.candidate_state is CandidateState.VERIFIED
                and state.binding_validity is not BindingValidity.VALID
            ):
                state.binding_validity = BindingValidity.VALID
                kinds.add(MilestoneKind.BINDING_BECAME_VALID)
        kinds.add(MilestoneKind.CHECKER_ACCEPTED)
    return kinds


def _record_math_result(
    state: _MutableState,
    arguments: dict[str, Any],
    response: dict[str, Any] | None,
) -> tuple[MilestoneKind, ...]:
    capability_id = arguments.get("capability_id")
    if not isinstance(capability_id, str):
        state.execution_status = "ERROR"
        return ()
    _record_binding(state, arguments)
    kinds: set[MilestoneKind] = set()
    if response is None:
        state.execution_status = "ERROR"
        return tuple(sorted(kinds, key=str))
    try:
        validated = CapabilityResult.model_validate(response)
    except ValidationError:
        state.execution_status = "ERROR"
        kinds.update(_record_diagnostics(state, response))
        return tuple(sorted(kinds, key=str))
    response = validated.model_dump(mode="json")
    if validated.capability_id != capability_id:
        if state.binding_validity is not BindingValidity.INVALID:
            state.binding_validity = BindingValidity.INVALID
            kinds.add(MilestoneKind.BINDING_BECAME_INVALID)
        state.execution_status = "ERROR"
        return tuple(sorted(kinds, key=str))
    execution = response.get("execution")
    status = execution.get("status") if isinstance(execution, dict) else None
    state.execution_status = status if isinstance(status, str) else "ERROR"
    kinds.update(_record_diagnostics(state, response))
    if state.execution_status in _NONCONCLUSIVE_EXECUTION:
        return tuple(sorted(kinds, key=str))
    output = response.get("output")
    kinds.update(_record_objects(state, capability_id, output))
    kinds.update(_record_artifacts(state, capability_id, response))
    kinds.update(_record_obligations(state, response))
    kinds.update(_record_scope_completeness_assurance(state, response))
    kinds.update(_record_checker(state, capability_id, validated))
    return tuple(sorted(kinds, key=str))


def _strict_events(raw: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrajectoryStateError(
                f"malformed Codex JSONL line {line_number}"
            ) from exc
        if not isinstance(event, dict):
            raise TrajectoryStateError(
                f"Codex JSONL line {line_number} is not an object"
            )
        events.append(event)
    return events


def _state_observation(
    event: dict[str, Any],
    state: _MutableState,
) -> tuple[StateBoundary, tuple[MilestoneKind, ...]] | None:
    if event.get("type") != "item.completed":
        return None
    item = event.get("item")
    if not isinstance(item, dict) or item.get("type") != "mcp_tool_call":
        return None
    if item.get("server") != _JACOBIAN_SERVER:
        return None
    tool = _tool_name(item.get("tool"))
    arguments = item.get("arguments")
    if tool is None or not isinstance(arguments, dict):
        return None
    response = _response_payload(item)
    if tool == "math.run":
        return StateBoundary.TOOL_RESULT, _record_math_result(
            state, arguments, response
        )
    # Only successful writes exist in the durable external log. A rejected
    # attempt is diagnostic tool traffic, not a state event.
    if item.get("status") != "completed" or response is None:
        return None
    boundary = _record_reasoning(state, arguments, response)
    return (boundary, ()) if boundary is not None else None


def extract_codex_trajectory(
    path: Path,
    *,
    task_family: str,
    terminal_evidence: CleanRoomTerminalEvidence | None = None,
) -> TrajectoryExtraction:
    """Extract versioned states from one immutable Codex JSONL transcript.

    Failed or malformed runtime results can change diagnostic state but do not
    become milestones.  Repeating an identical mathematical result cannot add
    an object, artifact, candidate, or transition and therefore receives no
    eligible milestone.
    """

    if path.is_symlink() or not path.is_file():
        raise TrajectoryStateError("trajectory source must be a regular file")
    raw = path.read_bytes()
    mutable = _MutableState(task_family)
    snapshots: list[ExtractedTrajectoryState] = []
    previous: TrajectoryHardState | None = None

    def append(
        boundary: StateBoundary,
        source_event_index: int,
        kinds: tuple[MilestoneKind, ...] = (),
    ) -> None:
        nonlocal previous
        mutable.latest_transitions = kinds
        hard = mutable.hard()
        changed = _changed_fields(previous, hard)
        eligible = bool(kinds)
        reason = (
            "eligible typed mathematical state transition: "
            + ", ".join(kind.value for kind in kinds)
            if eligible
            else "observation boundary only; no eligible typed mathematical state transition"
        )
        snapshots.append(
            ExtractedTrajectoryState(
                index=len(snapshots),
                source_event_index=source_event_index,
                boundary=boundary,
                hard_state=hard,
                soft_state=mutable.soft(),
                hard_state_digest=_digest(hard.model_dump(mode="json")),
                changed_fields=changed,
                milestone_kinds=kinds,
                milestone_eligible=eligible,
                milestone_reason=reason,
            )
        )
        previous = hard

    events = _strict_events(raw)
    for event_index, event in enumerate(events):
        observation = _state_observation(event, mutable)
        if observation is not None:
            boundary, kinds = observation
            append(boundary, event_index, kinds)

    if terminal_evidence is not None:
        before_binding = mutable.binding_validity
        bindings = (
            terminal_evidence.input_binding_valid,
            terminal_evidence.artifact_binding_valid,
        )
        if False in bindings:
            mutable.binding_validity = BindingValidity.INVALID
        elif all(value is True for value in bindings):
            mutable.binding_validity = BindingValidity.VALID
        terminal_kinds: tuple[MilestoneKind, ...] = ()
        if mutable.binding_validity != before_binding:
            # Terminal verifier binding is authoritative for the label, but it
            # is not intermediate credit and therefore remains non-milestone.
            mutable.latest_transitions = ()
        append(StateBoundary.TERMINAL, len(events), terminal_kinds)

    return TrajectoryExtraction(
        source_digest=_file_digest(raw),
        task_family=task_family,
        states=tuple(snapshots),
        terminal_evidence=terminal_evidence,
    )


__all__ = [
    "ArtifactStateRef",
    "BindingValidity",
    "CandidateState",
    "CheckerState",
    "CleanRoomTerminalEvidence",
    "ExtractedTrajectoryState",
    "MilestoneKind",
    "ReasoningProtocolState",
    "StateBoundary",
    "TerminalAcceptance",
    "TrajectoryExtraction",
    "TrajectoryHardState",
    "TrajectorySoftState",
    "TrajectoryStateError",
    "TypedObjectRef",
    "extract_codex_trajectory",
]
