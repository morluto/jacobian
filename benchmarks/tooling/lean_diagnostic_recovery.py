"""Operator-run Codex observation for Lean diagnostic-guided recovery."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
    model_validator,
)

from benchmarks.tooling.codex_visibility import (
    ToolMode,
    _codex_arguments,
    _command_version,
    _inspect_codex_skill_surface,
    _prepare_isolated_codex_environment,
    _sha256_bytes,
    _validate_mcp_url,
    inspect_surface,
    surface_snapshot_digest,
)
from benchmarks.tooling.command_runner import (
    ToolCommandStatus,
    git_head_sha,
    git_tracked_worktree_is_clean,
    run_operator_command,
)
from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.eval.telemetry import parse_agent_transcript, parse_agent_transcript_bytes

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SUITE = _ROOT / "benchmarks/config/lean-diagnostic-recovery-v1.json"
_REVISION = re.compile(r"^[0-9a-f]{12,40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHARED_REPORT_FIELDS = (
    "schema_version",
    "evidence_class",
    "causal_claim_authorized",
    "suite_id",
    "suite_digest",
    "source_base_revision",
    "source_candidate_revision",
    "model",
    "reasoning_effort",
    "tool_mode",
    "repetitions",
    "timeout_seconds",
    "codex_version",
    "evaluator",
    "selected_case_ids",
)
_DELTA_METRICS = (
    "repair_success_rate",
    "repeated_error_count",
    "math_run_call_count",
    "input_tokens",
    "output_tokens",
    "elapsed_seconds",
)
_OPERATIONAL_DIAGNOSTIC_CODES = frozenset(
    {
        "LEAN_CHECKER_TIMEOUT",
        "LEAN_MATHLIB_SETUP_FAILED",
        "LEAN_RUNTIME_SETUP_FAILED",
        "LEAN_TOOLCHAIN_SETUP_FAILED",
    }
)
_PROOF_DIAGNOSTIC_PHASES = frozenset(
    {
        "KERNEL_CHECK",
        "SOURCE_ELABORATION",
        "STATE_RECONSTRUCTION",
        "TACTIC_EXECUTION",
        "TERM_ELABORATION",
    }
)
_LEGACY_PROOF_DETAIL_PREFIXES = (
    "Lean proof has an unapproved trust base",
    "Lean rejected the proof",
)


class RecoveryCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Literal["control", "enriched-diagnostics"]
    description: str = Field(min_length=1)


class RecoveryDiagnosticEvidenceExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^(?:LEAN|INVALID_LEAN)_[A-Z0-9_]+$")
    path: str | None = None
    validation_error_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_unique_validation_paths(self) -> Self:
        if len(set(self.validation_error_paths)) != len(self.validation_error_paths):
            raise ValueError("expected validation-error paths must be unique")
        return self


class RecoveryDiagnosticProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_id: str = Field(min_length=1)
    payload: dict[str, Any]
    expected_diagnostic_evidence: RecoveryDiagnosticEvidenceExpectation


class RecoveryCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    injected_capability_id: str = Field(min_length=1)
    injected_payload: dict[str, Any]
    expected_diagnostic_codes: tuple[str, ...] = Field(min_length=1)
    expected_diagnostic_evidence: RecoveryDiagnosticEvidenceExpectation | None = None
    diagnostic_probe: RecoveryDiagnosticProbe | None = None
    terminal_capability_id: str = Field(min_length=1)
    terminal_immutable_input_fields: tuple[str, ...] = Field(min_length=1)
    prompt: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_stable_unique_codes(self) -> Self:
        if len(set(self.expected_diagnostic_codes)) != len(
            self.expected_diagnostic_codes
        ):
            raise ValueError("expected diagnostic codes must be unique")
        if any(
            not code.startswith(("LEAN_", "INVALID_LEAN_"))
            or not code.replace("_", "").isalnum()
            for code in self.expected_diagnostic_codes
        ):
            raise ValueError("expected diagnostic codes must be stable Lean codes")
        if (
            self.expected_diagnostic_evidence is not None
            and self.expected_diagnostic_evidence.code
            not in self.expected_diagnostic_codes
        ):
            raise ValueError(
                "expected diagnostic evidence code must be an expected diagnostic code"
            )
        if len(set(self.terminal_immutable_input_fields)) != len(
            self.terminal_immutable_input_fields
        ):
            raise ValueError("terminal immutable input fields must be unique")
        missing = (
            set(self.terminal_immutable_input_fields) - self.injected_payload.keys()
        )
        if missing:
            raise ValueError(
                "terminal immutable input fields must exist in the injected payload: "
                + ", ".join(sorted(missing))
            )
        return self


class RecoveryExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repetitions_per_case: int = Field(ge=1, le=20)
    timeout_seconds_per_rollout: float = Field(gt=0)
    wrong_answer_retries: Literal[0]
    web_search: Literal["disabled"]


class RecoverySuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    suite_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    evidence_class: str = Field(min_length=1)
    causal_claim_authorized: Literal[False]
    source_base_revision: str = Field(pattern=r"^[0-9a-f]{12,40}$")
    conditions: tuple[RecoveryCondition, RecoveryCondition]
    cases: tuple[RecoveryCase, ...] = Field(min_length=1)
    execution: RecoveryExecution
    primary_metric: Literal["repair_success_rate"]
    secondary_metrics: tuple[str, ...] = Field(min_length=1)
    invariants: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_cases_and_conditions(self) -> Self:
        if {condition.id for condition in self.conditions} != {
            "control",
            "enriched-diagnostics",
        }:
            raise ValueError("recovery study requires control and enriched conditions")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("recovery case IDs must be unique")
        return self


class RecoveryTokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    cached_input_tokens: StrictInt | None = Field(default=None, ge=0)
    cache_write_input_tokens: StrictInt | None = Field(default=None, ge=0)
    reasoning_output_tokens: StrictInt | None = Field(default=None, ge=0)


class RecoveryRunMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    injection_attempted: StrictBool
    injection_payload_exact: StrictBool
    injection_first_attempt: StrictBool
    injection_rejected: StrictBool
    observed_diagnostic_codes: tuple[str, ...]
    enriched_diagnostic_observed: StrictBool
    repair_success: StrictBool
    repeated_error_count: StrictInt = Field(ge=0)
    repeated_mcp_call_count: StrictInt = Field(ge=0)
    math_run_call_count: StrictInt = Field(ge=0)
    tool_error_count: StrictInt = Field(ge=0)
    tokens: RecoveryTokenUsage | None


class RecoveryRunCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ToolCommandStatus
    exit_code: StrictInt | None
    elapsed_seconds: StrictFloat = Field(ge=0, allow_inf_nan=False)


class RecoveryRunCommandReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ToolCommandStatus
    exit_code: StrictInt | None
    elapsed_microseconds: StrictInt = Field(ge=0)


class RecoveryRunArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str = Field(min_length=1)
    command_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    transcript: str = Field(min_length=1)
    transcript_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    stderr: str = Field(min_length=1)
    stderr_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class RetainedRecoveryRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    repetition: StrictInt = Field(ge=1)
    command: RecoveryRunCommand
    metrics: RecoveryRunMetrics
    artifacts: RecoveryRunArtifacts


def load_suite(path: Path) -> RecoverySuite:
    return RecoverySuite.model_validate_json(path.read_text(encoding="utf-8"))


def digest_suite(path: Path) -> str:
    """Bind an observation to the exact version-controlled suite bytes."""

    return _sha256_bytes(path.read_bytes())


def _candidate_revision(root: Path) -> str:
    if not git_tracked_worktree_is_clean(root):
        raise SystemExit(
            "recovery execution requires a clean tracked worktree; commit or "
            "stash evaluator changes before running"
        )
    revision = git_head_sha(root)
    if revision is None:
        raise SystemExit("cannot bind recovery report to the candidate Git revision")
    return revision


def _diagnostic_codes(invocation: Mapping[str, Any]) -> tuple[str, ...]:
    output = invocation.get("output")
    diagnostics = output.get("diagnostics") if isinstance(output, Mapping) else None
    if not isinstance(diagnostics, list):
        return ()
    return tuple(
        diagnostic["code"]
        for diagnostic in diagnostics
        if isinstance(diagnostic, Mapping) and isinstance(diagnostic.get("code"), str)
    )


def _invocation_diagnostics(
    invocation: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    output = invocation.get("output")
    diagnostics = output.get("diagnostics") if isinstance(output, Mapping) else None
    if not isinstance(diagnostics, list):
        return ()
    return tuple(value for value in diagnostics if isinstance(value, Mapping))


def _attempt_diagnostic_codes(attempt: Mapping[str, Any]) -> tuple[str, ...]:
    values = attempt.get("diagnostic_codes")
    if not isinstance(values, list):
        return ()
    return tuple(value for value in values if isinstance(value, str))


def _attempt_diagnostics(
    attempt: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    values = attempt.get("diagnostics")
    if not isinstance(values, list):
        return ()
    return tuple(value for value in values if isinstance(value, Mapping))


def _failed_request_rejection_codes(
    attempt: Mapping[str, Any],
) -> tuple[str, ...]:
    if attempt.get("successful") is True:
        return ()
    return tuple(
        code
        for code in _attempt_diagnostic_codes(attempt)
        if code.startswith("INVALID_LEAN_")
    )


def _diagnostic_evidence_observed(
    *,
    expected_codes: set[str],
    expectation: RecoveryDiagnosticEvidenceExpectation | None,
    codes: tuple[str, ...],
    diagnostics: tuple[Mapping[str, Any], ...],
) -> bool:
    if expectation is None:
        return bool(expected_codes & set(codes))
    if expectation.code not in codes:
        return False
    for diagnostic in diagnostics:
        if diagnostic.get("code") != expectation.code:
            continue
        if expectation.path is not None and diagnostic.get("path") != expectation.path:
            continue
        details = diagnostic.get("details")
        validation_errors = (
            details.get("validation_errors") if isinstance(details, Mapping) else None
        )
        if not isinstance(validation_errors, list):
            validation_errors = []
        observed_paths = {
            error.get("path")
            for error in validation_errors
            if isinstance(error, Mapping) and isinstance(error.get("path"), str)
        }
        if set(expectation.validation_error_paths).issubset(observed_paths):
            return True
    return False


def _injection_diagnostic_evidence_observed(
    case: RecoveryCase,
    *,
    codes: tuple[str, ...],
    diagnostics: tuple[Mapping[str, Any], ...],
) -> bool:
    return _diagnostic_evidence_observed(
        expected_codes=set(case.expected_diagnostic_codes),
        expectation=case.expected_diagnostic_evidence,
        codes=codes,
        diagnostics=diagnostics,
    )


def _diagnostic_probe_evidence_observed(
    case: RecoveryCase,
    attempts: tuple[Mapping[str, Any], ...],
) -> bool | None:
    probe = case.diagnostic_probe
    if probe is None:
        return None
    expectation = probe.expected_diagnostic_evidence
    return any(
        attempt.get("capability_id") == probe.capability_id
        and attempt.get("input") == probe.payload
        and _diagnostic_evidence_observed(
            expected_codes={expectation.code},
            expectation=expectation,
            codes=_attempt_diagnostic_codes(attempt),
            diagnostics=_attempt_diagnostics(attempt),
        )
        for attempt in attempts
    )


def _diagnostic_rejection_evidence(diagnostics: object) -> list[str]:
    evidence: list[str] = []
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if isinstance(diagnostic, str):
                if diagnostic.startswith(_LEGACY_PROOF_DETAIL_PREFIXES):
                    evidence.append(diagnostic)
                continue
            if not isinstance(diagnostic, Mapping):
                continue
            code = diagnostic.get("code")
            phase = diagnostic.get("phase")
            if (
                phase not in _PROOF_DIAGNOSTIC_PHASES
                or code in _OPERATIONAL_DIAGNOSTIC_CODES
            ):
                continue
            if isinstance(code, str):
                evidence.append(code)
    return evidence


def _input_rejection_evidence(input_validation: object) -> list[str]:
    if not isinstance(input_validation, Mapping):
        return []
    errors = input_validation.get("errors")
    if not isinstance(errors, list):
        return []
    return [
        error
        for error in errors
        if isinstance(error, str) and error.startswith(_LEGACY_PROOF_DETAIL_PREFIXES)
    ]


def _proof_rejection_evidence(invocation: Mapping[str, Any]) -> tuple[str, ...]:
    output = invocation.get("output")
    if not isinstance(output, Mapping):
        return ()
    evidence = _diagnostic_rejection_evidence(output.get("diagnostics"))
    input_validation = output.get("input")
    evidence.extend(_input_rejection_evidence(input_validation))
    if (
        "diagnostics" not in output
        and output.get("accepted") is False
        and output.get("baseline_accepted") is True
        and output.get("baseline_checker_execution_status") == "COMPLETED"
        and output.get("checker_execution_status") == "COMPLETED"
    ):
        # The legacy proof-edit contract did not project checker diagnostics. Its
        # accepted baseline proves that the same pinned runtime was available in
        # this atomic invocation before the edited proof was checked.
        evidence.append("LEGACY_PROOF_EDIT_REJECTION")
    return tuple(dict.fromkeys(evidence))


def _proof_invocation_rejected(invocation: Mapping[str, Any]) -> bool:
    output = invocation.get("output")
    if not isinstance(output, Mapping) or not _proof_rejection_evidence(invocation):
        return False
    input_validation = output.get("input")
    return bool(
        output.get("accepted") is False
        or output.get("conclusion") == "UNKNOWN"
        or (
            isinstance(input_validation, Mapping)
            and input_validation.get("status") == "REJECTED"
        )
    )


def _rejection_fingerprint(invocation: Mapping[str, Any]) -> tuple[str, str]:
    canonical_input = json.dumps(
        invocation.get("input"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return (
        str(invocation.get("capability_id")),
        _sha256_bytes(canonical_input),
    )


def _terminal_accepted(invocation: Mapping[str, Any]) -> bool:
    output = invocation.get("output")
    return bool(
        isinstance(output, Mapping)
        and (output.get("accepted") is True or output.get("conclusion") == "TRUE")
        and isinstance(invocation.get("verification_record_uri"), str)
    )


def _terminal_preserves_claim(
    case: RecoveryCase,
    invocation: Mapping[str, Any],
) -> bool:
    terminal_input = invocation.get("input")
    return bool(
        isinstance(terminal_input, Mapping)
        and all(
            field in terminal_input
            and terminal_input[field] == case.injected_payload[field]
            for field in case.terminal_immutable_input_fields
        )
    )


type _AttemptRecord = tuple[Mapping[str, Any], Mapping[str, Any] | None]


def _is_exact_injection(case: RecoveryCase, item: Mapping[str, Any]) -> bool:
    return bool(
        item.get("capability_id") == case.injected_capability_id
        and item.get("input") == case.injected_payload
    )


def _invocation_matches_attempt(
    attempt: Mapping[str, Any],
    invocation: Mapping[str, Any],
) -> bool:
    return bool(
        invocation.get("capability_id") == attempt.get("capability_id")
        and invocation.get("input") == attempt.get("input")
    )


def _pair_attempts_with_invocations(
    attempts: tuple[Mapping[str, Any], ...],
    invocations: tuple[Mapping[str, Any], ...],
) -> tuple[_AttemptRecord, ...]:
    records: list[_AttemptRecord] = []
    invocation_cursor = 0
    aligned = True
    for attempt in attempts:
        invocation = None
        if attempt.get("successful") is True:
            if invocation_cursor < len(invocations):
                candidate = invocations[invocation_cursor]
                if aligned and _invocation_matches_attempt(attempt, candidate):
                    invocation = candidate
                else:
                    # Attempts and completed invocations originate from the same
                    # ordered transcript. Once their identities diverge, their
                    # later positional relationship is ambiguous: do not shift a
                    # later proof rejection onto an earlier malformed response.
                    aligned = False
            invocation_cursor += 1
        records.append((attempt, invocation))
    return tuple(records)


def _qualifying_injection(
    case: RecoveryCase,
    records: tuple[_AttemptRecord, ...],
) -> tuple[int | None, tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    expected_codes = set(case.expected_diagnostic_codes)
    for index, (attempt, invocation) in enumerate(records):
        if not _is_exact_injection(case, attempt):
            continue
        if invocation is not None and _proof_invocation_rejected(invocation):
            return (
                index,
                _diagnostic_codes(invocation),
                _invocation_diagnostics(invocation),
            )
        request_codes = _failed_request_rejection_codes(attempt)
        if expected_codes & set(request_codes):
            return (
                index,
                _attempt_diagnostic_codes(attempt),
                _attempt_diagnostics(attempt),
            )
    return None, (), ()


def _repeated_rejection_count(records: tuple[_AttemptRecord, ...]) -> int:
    rejection_fingerprints: list[tuple[str, str]] = []
    for attempt, invocation in records:
        rejected = bool(
            (invocation is not None and _proof_invocation_rejected(invocation))
            or (invocation is None and _failed_request_rejection_codes(attempt))
        )
        if rejected:
            rejection_fingerprints.append(_rejection_fingerprint(attempt))
    return sum(
        fingerprint in rejection_fingerprints[:index]
        for index, fingerprint in enumerate(rejection_fingerprints)
    )


def classify_recovery(
    case: RecoveryCase,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    invocations = tuple(
        invocation
        for invocation in telemetry.get("capability_invocations", [])
        if isinstance(invocation, Mapping)
    )
    attempts = tuple(
        attempt
        for attempt in telemetry.get("capability_attempts", [])
        if isinstance(attempt, Mapping)
    )

    injection_payload_exact = any(
        _is_exact_injection(case, attempt) for attempt in attempts
    )
    injection_first_attempt = bool(attempts and _is_exact_injection(case, attempts[0]))
    attempt_records = _pair_attempts_with_invocations(attempts, invocations)
    qualifying_index, qualifying_codes, qualifying_diagnostics = _qualifying_injection(
        case, attempt_records
    )

    terminal = tuple(
        invocation
        for _attempt, invocation in (
            attempt_records[qualifying_index + 1 :]
            if qualifying_index is not None
            else ()
        )
        if invocation is not None
        and invocation.get("capability_id") == case.terminal_capability_id
    )
    repeated_errors = _repeated_rejection_count(attempt_records)
    probe_evidence = _diagnostic_probe_evidence_observed(case, attempts)
    usage = telemetry.get("usage")
    return {
        "injection_attempted": any(
            attempt.get("capability_id") == case.injected_capability_id
            for attempt in attempts
        ),
        "injection_payload_exact": injection_payload_exact,
        "injection_first_attempt": injection_first_attempt,
        "injection_rejected": qualifying_index is not None,
        "observed_diagnostic_codes": list(qualifying_codes),
        "enriched_diagnostic_observed": (
            probe_evidence
            if probe_evidence is not None
            else _injection_diagnostic_evidence_observed(
                case,
                codes=qualifying_codes,
                diagnostics=qualifying_diagnostics,
            )
        ),
        "repair_success": bool(
            qualifying_index is not None
            and any(
                _terminal_preserves_claim(case, invocation)
                and _terminal_accepted(invocation)
                for invocation in terminal
            )
        ),
        "repeated_error_count": repeated_errors,
        "repeated_mcp_call_count": int(telemetry.get("repeated_mcp_call_count", 0)),
        "math_run_call_count": sum(
            call == "math.run" for call in telemetry.get("mcp_calls", [])
        ),
        "tool_error_count": int(telemetry.get("tool_error_count", 0)),
        "tokens": usage if isinstance(usage, Mapping) else None,
    }


def _run_case(
    *,
    case: RecoveryCase,
    repetition: int,
    workspace: Path,
    output: Path,
    model: str,
    reasoning_effort: str,
    mcp_url: str,
    timeout_seconds: float,
    tool_mode: ToolMode,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    stem = f"{case.case_id}-r{repetition:02d}"
    command_path = output / f"{stem}.command.json"
    transcript_path = output / f"{stem}.jsonl"
    stderr_path = output / f"{stem}.stderr"
    started = time.monotonic()
    result = run_operator_command(
        "codex",
        _codex_arguments(
            workspace=workspace,
            model=model,
            reasoning_effort=reasoning_effort,
            mcp_url=mcp_url,
            prompt=case.prompt,
            tool_mode=tool_mode,
        ),
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        stdout_limit_bytes=16 * 1024 * 1024,
        stderr_limit_bytes=2 * 1024 * 1024,
        environment=environment,
    )
    transcript_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    telemetry = parse_agent_transcript(transcript_path)
    classified = classify_recovery(case, telemetry)
    command = RecoveryRunCommand(
        status=result.status,
        exit_code=result.exit_code,
        elapsed_seconds=round(time.monotonic() - started, 6),
    )
    command_receipt = RecoveryRunCommandReceipt(
        status=command.status,
        exit_code=command.exit_code,
        elapsed_microseconds=round(command.elapsed_seconds * 1_000_000),
    )
    command_payload = canonicalize_json(command_receipt.model_dump(mode="json"))
    command_path.write_bytes(command_payload)
    completed = command.status is ToolCommandStatus.EXITED and command.exit_code == 0
    return {
        "case_id": case.case_id,
        "repetition": repetition,
        "command": command.model_dump(mode="json"),
        "metrics": {
            **classified,
            "repair_success": completed and classified["repair_success"],
        },
        "artifacts": {
            "command": command_path.name,
            "command_sha256": _sha256_bytes(command_payload),
            "transcript": transcript_path.name,
            "transcript_sha256": _sha256_bytes(result.stdout),
            "stderr": stderr_path.name,
            "stderr_sha256": _sha256_bytes(result.stderr),
        },
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    run_count = len(runs)
    return {
        "run_count": run_count,
        "repair_success_count": sum(run["metrics"]["repair_success"] for run in runs),
        "repair_success_rate": (
            sum(run["metrics"]["repair_success"] for run in runs) / run_count
            if run_count
            else 0.0
        ),
        "enriched_diagnostic_observation_count": sum(
            run["metrics"]["enriched_diagnostic_observed"] for run in runs
        ),
        "injection_protocol_compliance_count": sum(
            run["metrics"]["injection_payload_exact"]
            and run["metrics"]["injection_rejected"]
            for run in runs
        ),
        "injection_first_attempt_count": sum(
            run["metrics"]["injection_first_attempt"] for run in runs
        ),
        "repeated_error_count": sum(
            run["metrics"]["repeated_error_count"] for run in runs
        ),
        "math_run_call_count": sum(
            run["metrics"]["math_run_call_count"] for run in runs
        ),
        "input_tokens": sum(
            (run["metrics"]["tokens"] or {}).get("input_tokens", 0) for run in runs
        ),
        "output_tokens": sum(
            (run["metrics"]["tokens"] or {}).get("output_tokens", 0) for run in runs
        ),
        "elapsed_seconds": round(
            sum(run["command"]["elapsed_seconds"] for run in runs), 6
        ),
    }


def _revision(report: Mapping[str, Any], field: str) -> str:
    value = report.get(field)
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise ValueError(f"recovery report requires a valid {field}")
    return value


def _same_revision(left: str, right: str) -> bool:
    return left.startswith(right) or right.startswith(left)


def _bind_observed_deployment_revision(
    surface: Mapping[str, Any],
    *,
    supplied_revision: str,
    expected_revision: str,
) -> str:
    deployment = surface.get("deployment")
    observed = deployment.get("revision") if isinstance(deployment, Mapping) else None
    if not isinstance(observed, str) or re.fullmatch(r"[0-9a-f]{40}", observed) is None:
        raise SystemExit("MCP endpoint omitted a canonical deployment revision")
    if not _same_revision(observed, supplied_revision):
        raise SystemExit(
            "observed MCP deployment revision does not match --deployed-revision"
        )
    if not _same_revision(observed, expected_revision):
        raise SystemExit(
            "observed MCP deployment revision does not match the condition source"
        )
    return observed


def _required_identity_strings(values: Mapping[str, Any]) -> dict[str, str]:
    identity: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(value, str) or not value:
            raise ValueError("recovery report has incomplete observed server identity")
        identity[key] = value
    return identity


def _deployment_surface_revision(
    deployment: object,
    *,
    server_version: str,
) -> str:
    if not isinstance(deployment, Mapping):
        raise ValueError("recovery report requires observed deployment metadata")
    revision = deployment.get("revision")
    if (
        deployment.get("schema_version") != "1"
        or deployment.get("evidence") != "release-marker"
        or deployment.get("package_version") != server_version
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise ValueError("recovery report has invalid observed deployment metadata")
    return revision


def _surface_identity(report: Mapping[str, Any]) -> dict[str, str]:
    surface = report.get("surface")
    if not isinstance(surface, Mapping):
        raise ValueError("recovery report requires an observed MCP surface")
    surface_digest = surface.get("surface_digest")
    server = surface.get("server")
    catalog = surface.get("catalog")
    if not isinstance(surface_digest, str) or _DIGEST.fullmatch(surface_digest) is None:
        raise ValueError("recovery report requires a valid surface digest")
    try:
        computed_surface_digest = surface_snapshot_digest(surface)
    except (TypeError, ValueError) as exc:
        raise ValueError("recovery report contains an invalid MCP surface") from exc
    if surface_digest != computed_surface_digest:
        raise ValueError("recovery report surface digest does not match its snapshot")
    if not isinstance(server, Mapping):
        raise ValueError("recovery report requires observed MCP server metadata")
    if not isinstance(catalog, Mapping):
        raise ValueError("recovery report requires observed catalog metadata")
    identity = _required_identity_strings(
        {
            "surface_digest": surface_digest,
            "server_name": server.get("name"),
            "server_version": server.get("version"),
            "catalog_digest": catalog.get("catalog_digest"),
            "policy_profile": catalog.get("policy_profile"),
            "policy_digest": catalog.get("policy_digest"),
        }
    )
    identity["deployment_revision"] = _deployment_surface_revision(
        surface.get("deployment"),
        server_version=identity["server_version"],
    )
    if (
        _DIGEST.fullmatch(identity["catalog_digest"]) is None
        or _DIGEST.fullmatch(identity["policy_digest"]) is None
    ):
        raise ValueError("recovery report has an invalid catalog or policy digest")
    return identity


def _validate_shared_report_invariants(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> None:
    if control.get("condition") != "control":
        raise ValueError("control report must use the control condition")
    if treatment.get("condition") != "enriched-diagnostics":
        raise ValueError("treatment report must use the enriched-diagnostics condition")
    _validate_evaluator_identity(control)
    _validate_evaluator_identity(treatment)
    for field in _SHARED_REPORT_FIELDS:
        if control.get(field) != treatment.get(field):
            raise ValueError(f"recovery comparison invariant differs: {field}")
    if control.get("schema_version") != "1":
        raise ValueError("recovery comparison requires report schema version 1")
    if control.get("causal_claim_authorized") is not False:
        raise ValueError("recovery reports cannot authorize causal claims")


def _validate_evaluator_identity(report: Mapping[str, Any]) -> None:
    evaluator = report.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("recovery reports require evaluator identity")
    isolation = evaluator.get("isolation")
    skill_surface = evaluator.get("skill_surface")
    if not isinstance(isolation, Mapping) or not isinstance(skill_surface, Mapping):
        raise ValueError(
            "recovery reports require evaluator isolation and skill surface"
        )
    if (
        isolation.get("home_isolated") is not True
        or isolation.get("codex_home_isolated") is not True
    ):
        raise ValueError("recovery reports require isolated evaluator homes")
    if skill_surface.get("external_file_sources") != []:
        raise ValueError("recovery evaluator skill surface is not isolated")
    digest = skill_surface.get("model_visible_instructions_sha256")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ValueError("recovery evaluator skill surface digest is invalid")


def _validate_selected_case_ids(report: Mapping[str, Any]) -> None:
    selected_case_ids = report.get("selected_case_ids")
    valid = (
        isinstance(selected_case_ids, list)
        and bool(selected_case_ids)
        and all(
            isinstance(case_id, str) and bool(case_id) for case_id in selected_case_ids
        )
        and len(set(selected_case_ids)) == len(selected_case_ids)
    )
    if not valid:
        raise ValueError("recovery reports require stable selected case IDs")


def _condition_bindings(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> tuple[str, str, dict[str, dict[str, str]]]:
    source_base_revision = _revision(control, "source_base_revision")
    source_candidate_revision = _revision(control, "source_candidate_revision")
    control_deployed_revision = _revision(control, "deployed_revision")
    treatment_deployed_revision = _revision(treatment, "deployed_revision")
    if not _same_revision(control_deployed_revision, source_base_revision):
        raise ValueError("control deployment does not match source_base_revision")
    if not _same_revision(treatment_deployed_revision, source_candidate_revision):
        raise ValueError(
            "treatment deployment does not match source_candidate_revision"
        )
    if _same_revision(control_deployed_revision, treatment_deployed_revision):
        raise ValueError("control and treatment must use different deployed revisions")
    control_identity = _surface_identity(control)
    treatment_identity = _surface_identity(treatment)
    if not _same_revision(
        control_identity["deployment_revision"], control_deployed_revision
    ):
        raise ValueError("control report deployment differs from observed MCP revision")
    if not _same_revision(
        treatment_identity["deployment_revision"], treatment_deployed_revision
    ):
        raise ValueError(
            "treatment report deployment differs from observed MCP revision"
        )
    if control_identity["surface_digest"] == treatment_identity["surface_digest"]:
        raise ValueError("control and treatment observed the same MCP surface")
    for field in ("server_name", "policy_profile", "policy_digest"):
        if control_identity[field] != treatment_identity[field]:
            raise ValueError(f"recovery server invariant differs: {field}")
    return (
        source_base_revision,
        source_candidate_revision,
        {
            "control": {
                "deployed_revision": control_deployed_revision,
                **control_identity,
            },
            "enriched-diagnostics": {
                "deployed_revision": treatment_deployed_revision,
                **treatment_identity,
            },
        },
    )


def _retained_runs(report: Mapping[str, Any]) -> list[RetainedRecoveryRun]:
    runs = report.get("runs")
    repetitions = report.get("repetitions")
    selected_case_ids = report.get("selected_case_ids")
    if (
        not isinstance(runs, list)
        or not runs
        or not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 1
        or not isinstance(selected_case_ids, list)
        or len(runs) != repetitions * len(selected_case_ids)
    ):
        raise ValueError("recovery report retained runs do not match its run plan")
    try:
        validated = [RetainedRecoveryRun.model_validate(run) for run in runs]
    except ValidationError as exc:
        raise ValueError("recovery report contains malformed retained runs") from exc
    expected = {
        (case_id, repetition)
        for case_id in selected_case_ids
        for repetition in range(1, repetitions + 1)
    }
    observed = [(run.case_id, run.repetition) for run in validated]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError(
            "recovery report must retain exactly one run per case and repetition"
        )
    artifact_names = [
        name
        for run in validated
        for name in (
            run.artifacts.command,
            run.artifacts.transcript,
            run.artifacts.stderr,
        )
    ]
    if len(set(artifact_names)) != len(artifact_names):
        raise ValueError("recovery report must retain distinct artifacts per run")
    return validated


def _artifact_path(report_root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or relative.name != name:
        raise ValueError("recovery run artifact names must be relative leaf names")
    try:
        resolved = (report_root / relative).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"recovery run artifact is unavailable: {name}") from exc
    if resolved.parent != report_root or not resolved.is_file():
        raise ValueError(f"recovery run artifact escapes its report directory: {name}")
    return resolved


def _verified_artifact(
    report_root: Path,
    *,
    name: str,
    expected_digest: str,
) -> tuple[Path, bytes]:
    path = _artifact_path(report_root, name)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"recovery run artifact cannot be read: {name}") from exc
    if _sha256_bytes(payload) != expected_digest:
        raise ValueError(f"recovery run artifact digest does not match: {name}")
    return path, payload


def _validate_transcript_jsonl(payload: bytes, *, name: str) -> None:
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
        events = [json.loads(line) for line in lines if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"recovery transcript is not valid JSONL: {name}") from exc
    if not events or any(not isinstance(event, Mapping) for event in events):
        raise ValueError(f"recovery transcript has no valid JSON events: {name}")


def _verified_command_receipt(
    run: RetainedRecoveryRun,
    *,
    report_root: Path,
) -> RecoveryRunCommand:
    _, payload = _verified_artifact(
        report_root,
        name=run.artifacts.command,
        expected_digest=run.artifacts.command_sha256,
    )
    try:
        receipt = RecoveryRunCommandReceipt.model_validate(loads_strict_json(payload))
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("recovery command receipt is malformed") from exc
    if canonicalize_json(receipt.model_dump(mode="json")) != payload:
        raise ValueError("recovery command receipt is not canonical")
    command = RecoveryRunCommand(
        status=receipt.status,
        exit_code=receipt.exit_code,
        elapsed_seconds=receipt.elapsed_microseconds / 1_000_000,
    )
    if command != run.command:
        raise ValueError("recovery command metadata does not match its receipt")
    return command


def _recomputed_run(
    run: RetainedRecoveryRun,
    *,
    case: RecoveryCase,
    report_root: Path,
) -> dict[str, Any]:
    command = _verified_command_receipt(run, report_root=report_root)
    _, transcript_payload = _verified_artifact(
        report_root,
        name=run.artifacts.transcript,
        expected_digest=run.artifacts.transcript_sha256,
    )
    _validate_transcript_jsonl(transcript_payload, name=run.artifacts.transcript)
    _verified_artifact(
        report_root,
        name=run.artifacts.stderr,
        expected_digest=run.artifacts.stderr_sha256,
    )
    try:
        telemetry = parse_agent_transcript_bytes(transcript_payload)
        classified = classify_recovery(case, telemetry)
        completed = (
            command.status is ToolCommandStatus.EXITED and command.exit_code == 0
        )
        metrics = RecoveryRunMetrics.model_validate(
            {
                **classified,
                "repair_success": completed and classified["repair_success"],
            }
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(
            f"recovery transcript cannot be classified for {run.case_id}"
        ) from exc
    retained_metrics = run.metrics.model_dump(mode="json", exclude_none=True)
    recomputed_metrics = metrics.model_dump(mode="json", exclude_none=True)
    if retained_metrics != recomputed_metrics:
        raise ValueError(
            "recovery retained metrics do not match the hash-verified transcript"
        )
    retained = run.model_dump(mode="json", exclude_none=True)
    retained["command"] = command.model_dump(mode="json", exclude_none=True)
    retained["metrics"] = recomputed_metrics
    return retained


def _summary(
    report: Mapping[str, Any],
    *,
    suite: RecoverySuite,
    report_root: Path,
) -> Mapping[str, Any]:
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("recovery reports require summaries")
    cases = {case.case_id: case for case in suite.cases}
    runs = _retained_runs(report)
    try:
        recomputed_runs = [
            _recomputed_run(run, case=cases[run.case_id], report_root=report_root)
            for run in runs
        ]
        computed = summarize_runs(recomputed_runs)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("recovery "):
            raise
        raise ValueError("recovery report contains malformed retained runs") from exc
    if summary != computed:
        raise ValueError("recovery report summary does not match retained runs")
    return computed


def _load_report(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[Path, Mapping[str, Any]]:
    if _DIGEST.fullmatch(expected_sha256) is None:
        raise ValueError("recovery report requires a valid external SHA-256 anchor")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("recovery report path is not a file")
        payload = resolved.read_bytes()
        if _sha256_bytes(payload) != expected_sha256:
            raise ValueError(
                "recovery report does not match its external SHA-256 anchor"
            )
        decoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"recovery report cannot be read: {path}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("recovery report must contain a JSON object")
    return resolved, decoded


def _validate_suite_binding(
    report: Mapping[str, Any],
    *,
    suite: RecoverySuite,
    suite_digest: str,
) -> None:
    if report.get("suite_id") != suite.suite_id:
        raise ValueError("recovery report suite_id does not match the selected suite")
    if report.get("suite_digest") != suite_digest:
        raise ValueError(
            "recovery report suite_digest does not match the selected suite"
        )
    if report.get("source_base_revision") != suite.source_base_revision:
        raise ValueError(
            "recovery report source_base_revision does not match the selected suite"
        )
    selected = report.get("selected_case_ids")
    suite_case_ids = {case.case_id for case in suite.cases}
    if not isinstance(selected, list) or not set(selected) <= suite_case_ids:
        raise ValueError("recovery report selects cases outside the selected suite")


def compare_report_paths(
    control_path: Path,
    treatment_path: Path,
    *,
    control_report_sha256: str,
    treatment_report_sha256: str,
    suite_path: Path = _DEFAULT_SUITE,
) -> dict[str, Any]:
    resolved_suite_path = suite_path.resolve(strict=True)
    suite = load_suite(resolved_suite_path)
    suite_digest = digest_suite(resolved_suite_path)
    resolved_control_path, control = _load_report(
        control_path,
        expected_sha256=control_report_sha256,
    )
    resolved_treatment_path, treatment = _load_report(
        treatment_path,
        expected_sha256=treatment_report_sha256,
    )
    _validate_shared_report_invariants(control, treatment)
    _validate_selected_case_ids(control)
    _validate_suite_binding(control, suite=suite, suite_digest=suite_digest)
    _validate_suite_binding(treatment, suite=suite, suite_digest=suite_digest)
    source_base_revision, source_candidate_revision, bindings = _condition_bindings(
        control, treatment
    )
    control_summary = _summary(
        control,
        suite=suite,
        report_root=resolved_control_path.parent,
    )
    treatment_summary = _summary(
        treatment,
        suite=suite,
        report_root=resolved_treatment_path.parent,
    )
    return {
        "schema_version": "1",
        "causal_claim_authorized": False,
        "suite_digest": control["suite_digest"],
        "source_base_revision": source_base_revision,
        "source_candidate_revision": source_candidate_revision,
        "control_condition": "control",
        "treatment_condition": "enriched-diagnostics",
        "report_sha256": {
            "control": control_report_sha256,
            "treatment": treatment_report_sha256,
        },
        "condition_bindings": bindings,
        "deltas": {
            metric: treatment_summary[metric] - control_summary[metric]
            for metric in _DELTA_METRICS
        },
        "interpretation": (
            "Descriptive public observation only; task identity and run invariants "
            "match, but this report does not authorize a causal claim."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--condition", choices=("control", "enriched-diagnostics"))
    parser.add_argument("--deployed-revision")
    parser.add_argument("--mcp-url")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--tool-mode", type=ToolMode, choices=tuple(ToolMode), default=ToolMode.DIRECT
    )
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--compare", nargs=2, type=Path, metavar=("CONTROL", "TREATMENT")
    )
    parser.add_argument("--control-report-sha256")
    parser.add_argument("--treatment-report-sha256")
    return parser


def _compare_from_arguments(args: argparse.Namespace) -> dict[str, Any]:
    if not args.control_report_sha256 or not args.treatment_report_sha256:
        raise SystemExit(
            "--compare requires externally retained --control-report-sha256 "
            "and --treatment-report-sha256 anchors"
        )
    return compare_report_paths(
        args.compare[0],
        args.compare[1],
        control_report_sha256=args.control_report_sha256,
        treatment_report_sha256=args.treatment_report_sha256,
        suite_path=args.suite,
    )


def _prepare_evaluator(
    isolated_root: Path,
    workspace: Path,
) -> tuple[Mapping[str, str], dict[str, Any], dict[str, Any]]:
    environment, isolation = _prepare_isolated_codex_environment(isolated_root)
    skill_surface = _inspect_codex_skill_surface(workspace, environment)
    if skill_surface["external_file_sources"]:
        raise RuntimeError("isolated Codex prompt exposed external file-backed skills")
    return environment, isolation, skill_surface


def main() -> None:
    args = _parser().parse_args()
    if args.compare:
        print(
            json.dumps(
                _compare_from_arguments(args),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not args.execute:
        raise SystemExit("refusing model execution without --execute")
    if (
        not args.condition
        or not args.deployed_revision
        or not args.mcp_url
        or not args.model
        or args.output is None
    ):
        raise SystemExit(
            "run mode requires --condition, --deployed-revision, --mcp-url, "
            "--model, and --output"
        )
    _validate_mcp_url(args.mcp_url)
    suite_path = args.suite.resolve(strict=True)
    suite = load_suite(suite_path)
    suite_digest = digest_suite(suite_path)
    source_candidate_revision = _candidate_revision(_ROOT)
    if _REVISION.fullmatch(args.deployed_revision) is None:
        raise SystemExit("--deployed-revision must be a 12- to 40-character Git SHA")
    expected_revision = (
        suite.source_base_revision
        if args.condition == "control"
        else source_candidate_revision
    )
    if not _same_revision(args.deployed_revision, expected_revision):
        raise SystemExit(
            f"{args.condition} deployment revision does not match {expected_revision}"
        )
    repetitions = args.repetitions or suite.execution.repetitions_per_case
    timeout = args.timeout_seconds or suite.execution.timeout_seconds_per_rollout
    if not 1 <= repetitions <= 20 or timeout <= 0:
        raise SystemExit("invalid repetition or timeout bound")
    available = {case.case_id for case in suite.cases}
    unknown = sorted(set(args.case) - available)
    if unknown:
        raise SystemExit(f"unknown case IDs: {unknown}")
    selected = tuple(
        case for case in suite.cases if not args.case or case.case_id in args.case
    )
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    surface = asyncio.run(
        inspect_surface(
            args.mcp_url,
            timeout,
            require_deployment_identity=True,
        )
    )
    observed_deployed_revision = _bind_observed_deployment_revision(
        surface,
        supplied_revision=args.deployed_revision,
        expected_revision=expected_revision,
    )
    output.mkdir(parents=True)
    with (
        tempfile.TemporaryDirectory(prefix="jacobian-lean-recovery-") as raw,
        tempfile.TemporaryDirectory(
            prefix="jacobian-lean-recovery-isolation-"
        ) as isolated,
    ):
        workspace = Path(raw)
        environment, isolation, skill_surface = _prepare_evaluator(
            Path(isolated), workspace
        )
        codex_version = _command_version(workspace, environment)
        runs = [
            _run_case(
                case=case,
                repetition=repetition,
                workspace=workspace,
                output=output,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                mcp_url=args.mcp_url,
                timeout_seconds=timeout,
                tool_mode=args.tool_mode,
                environment=environment,
            )
            for case in selected
            for repetition in range(1, repetitions + 1)
        ]
    if _candidate_revision(_ROOT) != source_candidate_revision:
        raise SystemExit("candidate Git revision changed during recovery execution")
    report = {
        "schema_version": "1",
        "evidence_class": suite.evidence_class,
        "causal_claim_authorized": False,
        "suite_id": suite.suite_id,
        "suite_digest": suite_digest,
        "source_base_revision": suite.source_base_revision,
        "source_candidate_revision": source_candidate_revision,
        "deployed_revision": observed_deployed_revision,
        "condition": args.condition,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "tool_mode": args.tool_mode,
        "repetitions": repetitions,
        "timeout_seconds": timeout,
        "codex_version": codex_version,
        "evaluator": {
            "isolation": isolation,
            "skill_surface": skill_surface,
        },
        "surface": surface,
        "selected_case_ids": [case.case_id for case in selected],
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    report_path = output / "report.json"
    report_payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    report_path.write_bytes(report_payload)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": _sha256_bytes(report_payload),
                "retention_required": "external-append-only-or-signed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
