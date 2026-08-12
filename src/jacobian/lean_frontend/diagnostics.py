"""Lean-owned conversion of backend messages into stable diagnostics."""

from __future__ import annotations

import re

from jacobian.contracts.lean import (
    LeanDiagnostic,
    LeanDiagnosticPhase,
    LeanDiagnosticPosition,
    LeanDiagnosticSource,
    LeanDiagnosticSourceSpan,
    LeanEnvironment,
)
from jacobian.contracts.results import VerificationResult
from jacobian.lean_frontend.artifacts import _PROOF_STATE_STATEMENT_PREFIX
from jacobian.lean_frontend.repl import _response_errors
from jacobian.lean_frontend.repl_protocol import (
    LeanReplErrorResponse,
    LeanReplMessage,
    LeanReplValidatedExecution,
)

_CHECKER_REJECTION = re.compile(
    r"^Lean rejected the proof at line (?P<line>\d+), column "
    r"(?P<column>\d+): (?P<message>.+?)\. Correct the proof body and retry\.$"
)
_METAVARIABLE = re.compile(r"\?m\.\d+|\?[A-Za-z_][A-Za-z0-9_.]*")
_INTERNAL_SCAFFOLD_WARNINGS = frozenset({"declaration uses `sorry`"})
_MAX_METAVARIABLE_LENGTH = 512
_MAX_RAW_BACKEND_MESSAGE_LENGTH = 20_000
_DiagnosticIdentity = tuple[
    LeanDiagnosticPhase,
    str,
    str,
    tuple[LeanDiagnosticSource, int, int, int, int] | None,
]
_OPERATIONAL_CHECKER_CLASSIFIERS = (
    (
        ("TOOLCHAIN_RESOLUTION:", "TOOLCHAIN_PROBE:"),
        "LEAN_TOOLCHAIN_SETUP_FAILED",
        "The pinned Lean toolchain is unavailable or is not authorized.",
    ),
    (
        ("MATHLIB_MANIFEST:",),
        "LEAN_MATHLIB_SETUP_FAILED",
        "The pinned Mathlib runtime failed manifest or package validation.",
    ),
    (
        ("LEAN_COMPILE_TIMEOUT:", "LEAN_CORE_TIMEOUT:"),
        "LEAN_CHECKER_TIMEOUT",
        "The Lean checker exceeded its bounded compile time.",
    ),
)
_MESSAGE_CLASSIFIERS = (
    (
        ("type mismatch", "application type mismatch"),
        "LEAN_TYPE_MISMATCH",
        "Lean reported a type mismatch.",
    ),
    (
        ("unsolved goals", "unsolved goal"),
        "LEAN_UNSOLVED_GOALS",
        "Lean left one or more goals unsolved.",
    ),
    (
        ("unknown identifier", "unknown constant"),
        "LEAN_UNKNOWN_IDENTIFIER",
        "Lean could not resolve an identifier.",
    ),
    (
        ("failed to synthesize", "failed to find instance"),
        "LEAN_SYNTHESIS_FAILED",
        "Lean could not synthesize a required instance or value.",
    ),
    (
        ("no goals to be solved",),
        "LEAN_NO_GOALS",
        "The tactic was applied after all goals were closed.",
    ),
    (
        ("forbidden lean command", "proof hole"),
        "LEAN_FORBIDDEN_SOURCE",
        "The source uses a forbidden Lean construct.",
    ),
)
_PHASE_FALLBACKS = {
    LeanDiagnosticPhase.RUNTIME_SETUP: (
        "LEAN_RUNTIME_SETUP_FAILED",
        "The Lean checker runtime could not be prepared.",
    ),
    LeanDiagnosticPhase.TERM_ELABORATION: (
        "LEAN_TERM_REJECTED",
        "Lean rejected the supplied term.",
    ),
    LeanDiagnosticPhase.TACTIC_EXECUTION: (
        "LEAN_TACTIC_REJECTED",
        "Lean rejected the supplied tactic.",
    ),
    LeanDiagnosticPhase.STATE_RECONSTRUCTION: (
        "LEAN_STATE_RECONSTRUCTION_FAILED",
        "Lean could not reconstruct the proof state.",
    ),
    LeanDiagnosticPhase.SOURCE_ELABORATION: (
        "LEAN_SOURCE_REJECTED",
        "Lean rejected the supplied source.",
    ),
    LeanDiagnosticPhase.KERNEL_CHECK: (
        "LEAN_PROOF_REJECTED",
        "Lean rejected the supplied proof.",
    ),
}


def repl_diagnostics(
    responses: LeanReplValidatedExecution,
    *,
    statement: str,
    final_phase: LeanDiagnosticPhase = LeanDiagnosticPhase.TACTIC_EXECUTION,
    final_source: LeanDiagnosticSource = LeanDiagnosticSource.TACTIC,
    final_column_offset: int = 0,
) -> tuple[LeanDiagnostic, ...]:
    """Convert one clean replay into bounded, payload-relative diagnostics."""

    diagnostics: list[LeanDiagnostic] = []
    seen: set[_DiagnosticIdentity] = set()
    for index, response in enumerate(responses):
        phase = (
            LeanDiagnosticPhase.SOURCE_ELABORATION
            if index == 0
            else (
                LeanDiagnosticPhase.STATE_RECONSTRUCTION if index == 1 else final_phase
            )
        )
        source = (
            LeanDiagnosticSource.STATEMENT
            if index == 0
            else (final_source if index == 2 else None)
        )
        offset = final_column_offset if index == 2 else 0
        if isinstance(response, LeanReplErrorResponse):
            _append_diagnostic(
                diagnostics,
                seen,
                raw=response.message,
                severity="ERROR",
                phase=phase,
                source=None,
                message=None,
                column_offset=0,
                goal_index=(0 if index == 2 else None),
                statement=None,
            )
            continue
        response_messages = response.messages
        for item in response_messages:
            if (
                index == 0
                and item.severity == "warning"
                and item.data.strip() in _INTERNAL_SCAFFOLD_WARNINGS
            ):
                # The initial source deliberately contains one private `sorry`
                # placeholder so the REPL can expose a proof state. It is not
                # part of the caller's statement, tactic, or term and therefore
                # must not outrank payload-owned diagnostics in agent results.
                continue
            severity = (
                "ERROR"
                if item.severity == "error"
                else ("WARNING" if item.severity == "warning" else "INFO")
            )
            _append_diagnostic(
                diagnostics,
                seen,
                raw=item.data,
                severity=severity,
                phase=phase,
                source=source,
                message=item,
                column_offset=offset,
                goal_index=(0 if index == 2 and severity == "ERROR" else None),
                statement=(statement if index == 0 else None),
            )
        emitted_errors = {item.data for item in response_messages}
        for raw in _response_errors(response):
            if raw in emitted_errors:
                continue
            _append_diagnostic(
                diagnostics,
                seen,
                raw=raw,
                severity="ERROR",
                phase=phase,
                source=None,
                message=None,
                column_offset=0,
                goal_index=(0 if index == 2 else None),
                statement=None,
            )
    return tuple(diagnostics)


def checker_diagnostics(
    result: VerificationResult,
    *,
    statement: str,
    proof: str,
    environment: LeanEnvironment,
) -> tuple[LeanDiagnostic, ...]:
    """Convert a checker rejection without changing its mathematical verdict."""

    diagnostics: list[LeanDiagnostic] = []
    for detail in result.input.errors:
        operational = _operational_checker_diagnostic(detail)
        if operational is not None:
            diagnostics.append(operational)
            continue
        match = _CHECKER_REJECTION.fullmatch(detail)
        raw = match.group("message") if match is not None else detail
        source_span = (
            _checker_payload_span(
                line=int(match.group("line")),
                column=int(match.group("column")),
                statement=statement,
                proof=proof,
                environment=environment,
            )
            if match is not None
            else None
        )
        code, message = _classify(raw, LeanDiagnosticPhase.KERNEL_CHECK)
        diagnostics.append(
            LeanDiagnostic(
                code=code,
                phase=LeanDiagnosticPhase.KERNEL_CHECK,
                severity="ERROR",
                message=message,
                source_span=source_span,
                metavariable=_first_metavariable(raw),
                raw_backend_message=_bounded_raw_backend_message(raw),
            )
        )
    return tuple(diagnostics)


def _operational_checker_diagnostic(detail: str) -> LeanDiagnostic | None:
    for prefixes, code, message in _OPERATIONAL_CHECKER_CLASSIFIERS:
        if detail.startswith(prefixes):
            return LeanDiagnostic(
                code=code,
                phase=LeanDiagnosticPhase.RUNTIME_SETUP,
                severity="ERROR",
                message=message,
                raw_backend_message=_bounded_raw_backend_message(detail),
            )
    if (detail.startswith("The pinned Lean ") and " executable " in detail) or (
        detail.startswith("Lean ") and " could not run locally." in detail
    ):
        return LeanDiagnostic(
            code="LEAN_RUNTIME_SETUP_FAILED",
            phase=LeanDiagnosticPhase.RUNTIME_SETUP,
            severity="ERROR",
            message="The Lean checker runtime could not be prepared.",
            raw_backend_message=_bounded_raw_backend_message(detail),
        )
    return None


def _append_diagnostic(
    diagnostics: list[LeanDiagnostic],
    seen: set[_DiagnosticIdentity],
    *,
    raw: str,
    severity: str,
    phase: LeanDiagnosticPhase,
    source: LeanDiagnosticSource | None,
    message: LeanReplMessage | None,
    column_offset: int,
    goal_index: int | None,
    statement: str | None,
) -> None:
    source_span = _repl_source_span(
        source,
        message,
        column_offset=column_offset,
        statement=statement,
    )
    span_identity = (
        (
            source_span.source,
            source_span.start.line,
            source_span.start.column,
            source_span.end.line,
            source_span.end.column,
        )
        if source_span is not None
        else None
    )
    key = (phase, severity, raw, span_identity)
    if key in seen:
        return
    seen.add(key)
    code, normalized = _classify(raw, phase)
    diagnostics.append(
        LeanDiagnostic.model_validate(
            {
                "code": code,
                "phase": phase,
                "severity": severity,
                "message": normalized,
                "source_span": source_span,
                "goal_index": goal_index,
                "metavariable": _first_metavariable(raw),
                "raw_backend_message": _bounded_raw_backend_message(raw),
            }
        )
    )


def _classify(
    raw: str,
    phase: LeanDiagnosticPhase,
) -> tuple[str, str]:
    lowered = raw.casefold()
    for needles, code, message in _MESSAGE_CLASSIFIERS:
        if any(needle in lowered for needle in needles):
            return code, message
    return _PHASE_FALLBACKS[phase]


def _repl_source_span(
    source: LeanDiagnosticSource | None,
    message: LeanReplMessage | None,
    *,
    column_offset: int,
    statement: str | None,
) -> LeanDiagnosticSourceSpan | None:
    if source is None or message is None:
        return None
    if source is LeanDiagnosticSource.STATEMENT:
        if statement is None:
            return None
        start_column = len(_PROOF_STATE_STATEMENT_PREFIX)
        end_column = start_column + len(statement)
        message_end = message.end_pos or message.pos
        if (
            message.pos.line != 0
            or message_end.line != 0
            or not start_column <= message.pos.column <= end_column
            or not start_column <= message_end.column <= end_column
        ):
            return None
        start = LeanDiagnosticPosition(
            line=0,
            column=message.pos.column - start_column,
        )
        end = LeanDiagnosticPosition(
            line=0,
            column=message_end.column - start_column,
        )
        if end.column < start.column:
            end = start
        return LeanDiagnosticSourceSpan(source=source, start=start, end=end)
    start = _offset_position(
        message.pos.line,
        message.pos.column,
        column_offset=column_offset,
    )
    end_position = message.end_pos or message.pos
    end = _offset_position(
        end_position.line,
        end_position.column,
        column_offset=column_offset,
    )
    if (end.line, end.column) < (start.line, start.column):
        end = start
    return LeanDiagnosticSourceSpan(source=source, start=start, end=end)


def _offset_position(
    line: int,
    column: int,
    *,
    column_offset: int,
) -> LeanDiagnosticPosition:
    return LeanDiagnosticPosition(
        line=line,
        column=max(0, column - column_offset) if line == 0 else column,
    )


def _checker_payload_span(
    *,
    line: int,
    column: int,
    statement: str,
    proof: str,
    environment: LeanEnvironment,
) -> LeanDiagnosticSourceSpan | None:
    theorem_line = 4 if environment is LeanEnvironment.MATHLIB else 3
    theorem_prefix = "theorem jacobian_theorem : ("
    proof_prefix = f"theorem jacobian_theorem : ({statement}) := "
    if line == theorem_line:
        if len(theorem_prefix) <= column <= len(theorem_prefix) + len(statement):
            position = LeanDiagnosticPosition(
                line=0,
                column=min(len(statement), max(0, column - len(theorem_prefix))),
            )
            source = LeanDiagnosticSource.STATEMENT
        elif column >= len(proof_prefix):
            position = LeanDiagnosticPosition(
                line=0,
                column=min(len(proof.splitlines()[0]), column - len(proof_prefix)),
            )
            source = LeanDiagnosticSource.PROOF
        else:
            return None
    elif line > theorem_line:
        complete_proof_term = re.match(r"^by(?:\s|$)", proof.lstrip()) is not None
        proof_line = (
            line - theorem_line if complete_proof_term else line - theorem_line - 1
        )
        proof_lines = proof.splitlines()
        if proof_line < 0 or proof_line >= len(proof_lines):
            return None
        position = LeanDiagnosticPosition(
            line=proof_line,
            column=min(
                len(proof_lines[proof_line]),
                max(0, column if complete_proof_term else column - 2),
            ),
        )
        source = LeanDiagnosticSource.PROOF
    else:
        return None
    return LeanDiagnosticSourceSpan(source=source, start=position, end=position)


def _first_metavariable(raw: str) -> str | None:
    match = _METAVARIABLE.search(raw)
    return match.group(0)[:_MAX_METAVARIABLE_LENGTH] if match is not None else None


def _bounded_raw_backend_message(raw: str) -> str:
    if not raw:
        return "Lean returned an empty diagnostic."
    return raw[:_MAX_RAW_BACKEND_MESSAGE_LENGTH]


__all__ = ["checker_diagnostics", "repl_diagnostics"]
