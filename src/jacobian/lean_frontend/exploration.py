"""Pinned, replayable exploratory Lean capabilities."""

from __future__ import annotations

import re
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.capabilities import (
    CapabilityProviderRuntime,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_exploration import (
    LeanPremiseRetrievalArtifact,
    LeanProofStateArtifact,
    LeanProofStateRequest,
    LeanProofStateTransitionArtifact,
    LeanTacticDiagnostic,
    LeanTypedGoal,
)
from jacobian.contracts.lean_metavariable_fields import (
    LeanMetavariableFieldsArtifact,
    LeanMetavariableFieldsRequest,
)
from jacobian.lean_frontend.repl import (
    LeanExplorationReplRuntime,
    _response_errors,
)
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.references import LeanCheckerInstallation
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.worker_environment import worker_environment


class LeanHelperError(RuntimeError):
    """Carries a specific error code from the pinned Lean helper."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


if TYPE_CHECKING:
    from jacobian.lean_frontend.metavariable_fields import (
        LeanMetavariableFieldsAdapter,
    )
    from jacobian.lean_frontend.premise_retrieval import LeanPremiseRetrievalAdapter
    from jacobian.lean_frontend.proof_state import LeanProofStateAdapter
    from jacobian.lean_frontend.proof_state_inspect import (
        LeanProofStateInspectAdapter,
    )
    from jacobian.lean_frontend.term_apply import LeanTermApplyAdapter

_FORBIDDEN = re.compile(
    r"\b(?:admit|axiom|elab|import|macro|native_decide|opaque|run_tac|"
    r"set_option|sorry|syntax|unsafe)\b|#",
    re.IGNORECASE,
)
_SUGGESTION = re.compile(
    r"Try this:\s*\n\s*\[apply\]\s*(?P<tactic>[^\r\n]+)",
)
_DECLARATION = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_']+)+\b")
_RESOURCE_POLL_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class LeanExplorationInstallation:
    semantics_uri: str
    state_schema_uri: str
    transition_schema_uri: str
    retrieval_schema_uri: str
    metavariable_schema_uri: str
    repl: LeanExplorationReplRuntime


@dataclass(frozen=True, slots=True)
class _Resources:
    store: ArtifactRepository
    artifacts: ArtifactService
    semantics_uri: str
    state_schema_uri: str
    transition_schema_uri: str
    retrieval_schema_uri: str
    installations: Mapping[LeanEnvironment, LeanCheckerInstallation]
    runtime: Path
    provider_runtime: CapabilityProviderRuntime
    repl: LeanExplorationReplRuntime


def install_lean_exploration_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    installations: Mapping[LeanEnvironment, LeanCheckerInstallation],
    provider_runtime: CapabilityProviderRuntime,
) -> tuple[
    tuple[
        LeanProofStateAdapter,
        LeanPremiseRetrievalAdapter,
        LeanTermApplyAdapter,
        LeanProofStateInspectAdapter,
        LeanMetavariableFieldsAdapter,
    ],
    LeanExplorationInstallation,
]:
    """Register replayable exploratory Lean adapters."""

    mathlib = installations[LeanEnvironment.MATHLIB]
    core = installations[LeanEnvironment.CORE]
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.lean4-exploration",
        version="1",
        definition={
            "description": (
                "immutable replayable Lean proof states, one-step tactic "
                "transitions, and premise suggestions"
            ),
            "lean_version": core.lean_version,
            "lean_commit": core.lean_commit,
            "mathlib_commit": mathlib.mathlib_commit,
            "state_expiry": "immutable artifacts do not expire",
            "verification": "none; completed source must pass lean.check",
        },
    )
    state_schema_uri = schemas.register(
        name="jacobian.lean4-proof-state",
        version="1",
        schema=LeanProofStateArtifact.model_json_schema(),
    )
    transition_schema_uri = schemas.register(
        name="jacobian.lean4-proof-state-transition",
        version="2",
        schema=LeanProofStateTransitionArtifact.model_json_schema(),
    )
    retrieval_schema_uri = schemas.register(
        name="jacobian.lean4-premise-retrieval",
        version="2",
        schema=LeanPremiseRetrievalArtifact.model_json_schema(),
    )
    metavariable_schema_uri = schemas.register(
        name="jacobian.lean4-metavariable-fields",
        version="1",
        schema=LeanMetavariableFieldsArtifact.model_json_schema(),
    )
    runtime = Path(__file__).resolve().parents[3] / "lean"
    repl = LeanExplorationReplRuntime(runtime, installations)
    resources = _Resources(
        store=store,
        artifacts=artifacts,
        semantics_uri=semantics_uri,
        state_schema_uri=state_schema_uri,
        transition_schema_uri=transition_schema_uri,
        retrieval_schema_uri=retrieval_schema_uri,
        installations=installations,
        runtime=runtime,
        provider_runtime=provider_runtime,
        repl=repl,
    )
    from jacobian.lean_frontend.metavariable_fields import (
        LeanMetavariableFieldsAdapter,
    )
    from jacobian.lean_frontend.premise_retrieval import LeanPremiseRetrievalAdapter
    from jacobian.lean_frontend.proof_state import LeanProofStateAdapter
    from jacobian.lean_frontend.proof_state_inspect import (
        LeanProofStateInspectAdapter,
    )
    from jacobian.lean_frontend.term_apply import LeanTermApplyAdapter

    proof_state_adapter = LeanProofStateAdapter(resources)
    return (
        (
            proof_state_adapter,
            LeanPremiseRetrievalAdapter(resources),
            LeanTermApplyAdapter(proof_state_adapter, provider_runtime),
            LeanProofStateInspectAdapter(resources, provider_runtime),
            LeanMetavariableFieldsAdapter(
                resources, metavariable_schema_uri, provider_runtime
            ),
        ),
        LeanExplorationInstallation(
            semantics_uri=semantics_uri,
            state_schema_uri=state_schema_uri,
            transition_schema_uri=transition_schema_uri,
            retrieval_schema_uri=retrieval_schema_uri,
            metavariable_schema_uri=metavariable_schema_uri,
            repl=repl,
        ),
    )


def _validate_source_parts(statement: str, tactics: tuple[str, ...]) -> None:
    if "\n" in statement or "\r" in statement or ":=" in statement:
        raise ValueError("statement must be one Lean expression")
    if _FORBIDDEN.search(statement):
        raise ValueError("statement contains a forbidden command")
    for tactic in tactics:
        if "\x00" in tactic or _FORBIDDEN.search(tactic):
            raise ValueError("tactic contains a forbidden command")


def _normalized_response_goals(response: Mapping[str, Any]) -> tuple[str, ...]:
    goals_value = response.get("goals", [])
    if not isinstance(goals_value, list) or any(
        not isinstance(goal, str) for goal in goals_value
    ):
        raise RuntimeError("Lean REPL returned malformed goals")
    return tuple(_normalize_goal(goal) for goal in goals_value)


def _normalize_goal(goal: str) -> str:
    lines = goal.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(line.rstrip() for line in lines).strip()


def _tactic_diagnostics(
    responses: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> tuple[LeanTacticDiagnostic, ...]:
    diagnostics: list[LeanTacticDiagnostic] = []
    for response in responses:
        seen: set[str] = set()
        for message in _response_errors(response):
            if message in seen:
                continue
            seen.add(message)
            diagnostics.append(LeanTacticDiagnostic(severity="ERROR", message=message))
        structured = response.get("messages")
        if not isinstance(structured, list):
            continue
        for item in structured:
            if not isinstance(item, Mapping):
                continue
            data = item.get("data")
            if not isinstance(data, str):
                continue
            if data in seen:
                continue
            seen.add(data)
            raw_severity = item.get("severity")
            severity = (
                "ERROR"
                if raw_severity == "error"
                else ("WARNING" if raw_severity == "warning" else "INFO")
            )
            diagnostics.append(
                LeanTacticDiagnostic.model_validate(
                    {"severity": severity, "message": data}
                )
            )
    return tuple(diagnostics)


def _run_repl(
    resources: _Resources,
    *,
    command: str,
    tactic: str,
    environment: LeanEnvironment,
    pickle_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return resources.repl.execute(
        command=command,
        tactic=tactic,
        environment=environment,
        pickle_path=pickle_path,
    )


def _resolve_typed_goal_helper(
    resources: _Resources,
    request: LeanProofStateRequest,
    query_path: Path,
) -> tuple[str, tuple[str, ...], dict[str, str]]:
    """Resolve the typed-goal helper command and environment."""

    helper = resources.runtime / ".lake" / "build" / "bin" / "jacobian_lean_proof_state"
    if not helper.is_file():
        raise RuntimeError(
            "the pinned typed proof-state helper is unavailable; "
            "run `lake build jacobian_lean_proof_state` in lean/"
        )
    elan = shutil.which("elan")
    if elan is None:
        raise RuntimeError("elan is unavailable")
    installation = resources.installations[request.environment]
    environment = worker_environment(
        extra_variables=("HOME", "PATH", "ELAN_HOME"),
        overrides={"JACOBIAN_LEAN_PROOF_STATE_QUERY": str(query_path)},
    )
    arguments = (
        "run",
        f"leanprover/lean4:v{installation.lean_version}",
        "lake",
        "env",
        str(helper),
    )
    return elan, arguments, environment


def _parse_typed_goal_envelope(
    stdout: bytes,
    *,
    request_id: str,
) -> dict[str, Any]:
    """Extract and validate the typed-goal JSON envelope from helper output.

    If the helper emitted an error envelope (``JACOBIAN_PROOF_STATE_ERROR``),
    raise :class:`LeanHelperError` carrying the helper's specific error code
    instead of collapsing to a generic parse failure.
    """

    result_marker = "JACOBIAN_PROOF_STATE_RESULT "
    error_marker = "JACOBIAN_PROOF_STATE_ERROR "
    try:
        lines = stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("Lean typed proof-state extraction failed") from exc
    error_lines = [line for line in lines if line.startswith(error_marker)]
    if error_lines:
        try:
            error_envelope = loads_strict_json(
                error_lines[0].removeprefix(error_marker)
            )
        except CanonicalizationError as exc:
            raise RuntimeError("Lean typed proof-state extraction failed") from exc
        if isinstance(error_envelope, dict):
            code = error_envelope.get("code")
            message = error_envelope.get("message", "")
            envelope_request_id = error_envelope.get("request_id")
            if (
                isinstance(code, str)
                and isinstance(message, str)
                and isinstance(envelope_request_id, str)
                and envelope_request_id == request_id
            ):
                raise LeanHelperError(code, message)
        raise RuntimeError("Lean typed proof-state extraction failed")
    responses = [line for line in lines if line.startswith(result_marker)]
    if len(responses) != 1:
        raise RuntimeError("Lean typed proof-state extraction failed")
    try:
        envelope = loads_strict_json(responses[0].removeprefix(result_marker))
    except CanonicalizationError as exc:
        raise RuntimeError("Lean typed proof-state extraction failed") from exc
    if not isinstance(envelope, dict) or envelope.get("request_id") != request_id:
        raise RuntimeError("Lean typed proof-state extraction returned invalid JSON")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("Lean typed proof-state extraction returned invalid JSON")
    return payload


def _extract_typed_goals(
    resources: _Resources,
    *,
    pickle_path: Path,
    request: LeanProofStateRequest,
) -> tuple[LeanTypedGoal, ...]:
    query_path = pickle_path.with_name("typed-goal-query.json")
    request_id = uuid.uuid4().hex
    query_path.write_bytes(
        canonicalize_json(
            {
                "pickle_path": str(pickle_path),
                "request_id": request_id,
                "max_goals": request.max_goals,
                "max_local_declarations": request.max_local_declarations,
                "max_rendered_bytes": request.max_rendered_bytes,
                "mode": "typed_goals",
            }
        )
    )
    elan, arguments, environment = _resolve_typed_goal_helper(
        resources, request, query_path
    )
    try:
        result = execute_process(
            ProcessRequest(
                executable=elan,
                arguments=arguments,
                environment=environment,
                cwd=str(resources.runtime),
                timeout_seconds=30.0,
                stdin_bytes=b"",
                stdout_limit_bytes=2 * 1024 * 1024,
                stderr_limit_bytes=128 * 1024,
            )
        )
    except OSError as exc:
        raise RuntimeError("Lean typed proof-state extraction failed") from exc
    if result.termination is not ProcessTermination.EXITED:
        raise RuntimeError("Lean typed proof-state extraction failed")
    if result.returncode != 0:
        raise RuntimeError("Lean typed proof-state extraction failed")
    payload = _parse_typed_goal_envelope(result.stdout, request_id=request_id)
    if payload.get("expression_serialization") != "LEAN_PRETTY_PRINTED_EXPR":
        raise RuntimeError("Lean typed proof-state serialization is unsupported")
    try:
        return tuple(
            LeanTypedGoal.model_validate(goal) for goal in payload["typed_goals"]
        )
    except (KeyError, TypeError, ValidationError) as exc:
        raise RuntimeError(
            "Lean typed proof-state extraction returned invalid goals"
        ) from exc


def _extract_structured_metavariables(
    resources: _Resources,
    *,
    pickle_path: Path,
    request: LeanMetavariableFieldsRequest,
) -> dict[str, Any]:
    """Extract structured metavariable fields via the pinned Lean helper.

    The helper is invoked with ``mode = "metavariable_fields"`` so it emits
    ``structured_metavariables``, ``elaboration_context``, and the honest
    ``coercion_provenance`` field through maintained Lean accessors rather
    than by parsing pretty-printed output.
    """

    query_path = pickle_path.with_name("metavariable-fields-query.json")
    request_id = uuid.uuid4().hex
    query_path.write_bytes(
        canonicalize_json(
            {
                "pickle_path": str(pickle_path),
                "request_id": request_id,
                "max_goals": request.max_goals,
                "max_local_declarations": request.max_local_declarations,
                "max_rendered_bytes": request.max_rendered_bytes,
                "mode": "metavariable_fields",
            }
        )
    )
    helper_request = LeanProofStateRequest.model_validate(
        {
            "environment": request.environment.value,
            "statement": "True",
            "tactic": "skip",
            "max_goals": request.max_goals,
            "max_local_declarations": request.max_local_declarations,
            "max_rendered_bytes": request.max_rendered_bytes,
        }
    )
    elan, arguments, environment = _resolve_typed_goal_helper(
        resources, helper_request, query_path
    )
    try:
        result = execute_process(
            ProcessRequest(
                executable=elan,
                arguments=arguments,
                environment=environment,
                cwd=str(resources.runtime),
                timeout_seconds=30.0,
                stdin_bytes=b"",
                stdout_limit_bytes=2 * 1024 * 1024,
                stderr_limit_bytes=128 * 1024,
            )
        )
    except OSError as exc:
        raise RuntimeError("Lean metavariable-field extraction failed") from exc
    if result.termination is not ProcessTermination.EXITED:
        raise RuntimeError("Lean metavariable-field extraction failed")
    if result.returncode != 0:
        raise RuntimeError("Lean metavariable-field extraction failed")
    payload = _parse_typed_goal_envelope(result.stdout, request_id=request_id)
    if payload.get("expression_serialization") != "LEAN_PRETTY_PRINTED_EXPR":
        raise RuntimeError("Lean metavariable-field serialization is unsupported")
    if payload.get("coercion_provenance") != "UNAVAILABLE":
        raise RuntimeError("Lean metavariable-field coercion provenance is invalid")
    return payload


def _response_messages(response: Mapping[str, Any]) -> tuple[str, ...]:
    messages: list[str] = []
    message = response.get("message")
    if isinstance(message, str):
        messages.append(message)
    structured = response.get("messages")
    if isinstance(structured, list):
        for item in structured:
            if not isinstance(item, Mapping):
                continue
            data = item.get("data")
            if isinstance(data, str):
                messages.append(data)
    return tuple(messages)


def _runtime_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
