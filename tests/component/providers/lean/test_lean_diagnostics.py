from __future__ import annotations

import pytest

from jacobian.contracts.lean import (
    LeanDiagnosticPhase,
    LeanDiagnosticSource,
    LeanEnvironment,
)
from jacobian.contracts.results import VerificationResult
from jacobian.lean_frontend.diagnostics import checker_diagnostics, repl_diagnostics
from jacobian.lean_frontend.repl_protocol import (
    LeanReplCommandResponse,
    LeanReplProofStepResponse,
)


def _command_with_warning(message: str) -> LeanReplCommandResponse:
    return LeanReplCommandResponse.model_validate(
        {
            "env": 0,
            "messages": [
                {
                    "pos": {"line": 1, "column": 0},
                    "endPos": {"line": 1, "column": 7},
                    "severity": "warning",
                    "data": message,
                }
            ],
            "sorries": [{"goal": "⊢ True", "proofState": 0}],
        }
    )


def _proof_step(*, error: str | None = None) -> LeanReplProofStepResponse:
    payload: dict[str, object] = {
        "proofState": 1,
        "proofStatus": "Goals",
        "goals": ["⊢ True"],
    }
    if error is not None:
        payload["messages"] = [
            {
                "pos": {"line": 0, "column": 6},
                "endPos": {"line": 0, "column": 7},
                "severity": "error",
                "data": error,
            }
        ]
    return LeanReplProofStepResponse.model_validate(payload)


def _rejected_checker_result(detail: str) -> VerificationResult:
    return VerificationResult.model_validate(
        {
            "execution": {"status": "COMPLETED"},
            "input": {"status": "REJECTED", "errors": [detail]},
            "conclusion": "UNKNOWN",
        }
    )


def test_repl_diagnostics_omit_the_private_sorry_scaffold_warning() -> None:
    diagnostics = repl_diagnostics(
        (
            _command_with_warning("declaration uses `sorry`"),
            _proof_step(),
            _proof_step(error="type mismatch"),
        ),
        statement="True",
        final_phase=LeanDiagnosticPhase.TERM_ELABORATION,
        final_source=LeanDiagnosticSource.TERM,
        final_column_offset=len("exact "),
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "LEAN_TYPE_MISMATCH"
    assert diagnostics[0].phase is LeanDiagnosticPhase.TERM_ELABORATION
    assert diagnostics[0].source_span is not None
    assert diagnostics[0].source_span.source is LeanDiagnosticSource.TERM


def test_repl_diagnostics_keep_other_source_warnings() -> None:
    diagnostics = repl_diagnostics(
        (
            _command_with_warning("caller-visible source warning"),
            _proof_step(),
            _proof_step(),
        ),
        statement="True",
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].phase is LeanDiagnosticPhase.SOURCE_ELABORATION
    assert diagnostics[0].severity == "WARNING"
    assert diagnostics[0].raw_backend_message == "caller-visible source warning"
    assert diagnostics[0].source_span is None


def test_repl_diagnostics_bound_backend_text_and_metavariable() -> None:
    raw = "type mismatch ?" + "a" * 25_000

    diagnostics = repl_diagnostics(
        (
            _command_with_warning("declaration uses `sorry`"),
            _proof_step(),
            _proof_step(error=raw),
        ),
        statement="True",
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "LEAN_TYPE_MISMATCH"
    assert len(diagnostics[0].raw_backend_message) == 20_000
    assert diagnostics[0].raw_backend_message == raw[:20_000]
    assert diagnostics[0].metavariable is not None
    assert len(diagnostics[0].metavariable) == 512


def test_repl_diagnostics_translate_generated_statement_columns() -> None:
    prefix_length = len("example : ")
    command = LeanReplCommandResponse.model_validate(
        {
            "env": 0,
            "messages": [
                {
                    "pos": {"line": 0, "column": prefix_length + 1},
                    "endPos": {"line": 0, "column": prefix_length + 4},
                    "severity": "warning",
                    "data": "caller-visible statement warning",
                }
            ],
            "sorries": [{"goal": "⊢ True", "proofState": 0}],
        }
    )

    diagnostics = repl_diagnostics(
        (command, _proof_step(), _proof_step()),
        statement="True",
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].source_span is not None
    assert diagnostics[0].source_span.source is LeanDiagnosticSource.STATEMENT
    assert diagnostics[0].source_span.start.line == 0
    assert diagnostics[0].source_span.start.column == 1
    assert diagnostics[0].source_span.end.column == 4


def test_repl_diagnostics_preserve_identical_messages_at_distinct_spans() -> None:
    prefix_length = len("example : ")
    command = LeanReplCommandResponse.model_validate(
        {
            "env": 0,
            "messages": [
                {
                    "pos": {"line": 0, "column": prefix_length},
                    "endPos": {"line": 0, "column": prefix_length + 4},
                    "severity": "error",
                    "data": "unknown identifier",
                },
                {
                    "pos": {"line": 0, "column": prefix_length + 7},
                    "endPos": {"line": 0, "column": prefix_length + 11},
                    "severity": "error",
                    "data": "unknown identifier",
                },
            ],
            "sorries": [{"goal": "⊢ True ∧ True", "proofState": 0}],
        }
    )

    diagnostics = repl_diagnostics(
        (command, _proof_step(), _proof_step()),
        statement="True ∧ True",
    )

    assert len(diagnostics) == 2
    spans = [diagnostic.source_span for diagnostic in diagnostics]
    assert all(span is not None for span in spans)
    assert [span.start.column for span in spans if span is not None] == [0, 7]


def test_checker_diagnostics_classify_setup_failure_as_operational() -> None:
    detail = (
        "MATHLIB_MANIFEST: a pinned mathlib package checkout failed integrity "
        "validation"
    )

    diagnostics = checker_diagnostics(
        _rejected_checker_result(detail),
        statement="True",
        proof="by trivial",
        environment=LeanEnvironment.MATHLIB,
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].code == "LEAN_MATHLIB_SETUP_FAILED"
    assert diagnostics[0].phase is LeanDiagnosticPhase.RUNTIME_SETUP
    assert diagnostics[0].source_span is None
    assert diagnostics[0].raw_backend_message == detail


def test_checker_diagnostics_keep_toolchain_failure_out_of_proof_repair() -> None:
    detail = (
        "TOOLCHAIN_PROBE: The pinned Lean 4.31.0 toolchain is unavailable. "
        "Install it and retry."
    )

    diagnostics = checker_diagnostics(
        _rejected_checker_result(detail),
        statement="True",
        proof="by trivial",
        environment=LeanEnvironment.CORE,
    )

    assert [diagnostic.code for diagnostic in diagnostics] == [
        "LEAN_TOOLCHAIN_SETUP_FAILED"
    ]
    assert all(
        diagnostic.phase is LeanDiagnosticPhase.RUNTIME_SETUP
        for diagnostic in diagnostics
    )


@pytest.mark.parametrize(
    "detail",
    (
        "The pinned Lean 4.31.0 executable could not be resolved.",
        "The pinned Lean 4.31.0 executable is unavailable.",
        (
            "The pinned Lean 4.31.0 executable resolved to the elan proxy rather "
            "than the toolchain binary."
        ),
    ),
)
def test_checker_diagnostics_classify_all_executable_resolution_failures(
    detail: str,
) -> None:
    diagnostics = checker_diagnostics(
        _rejected_checker_result(detail),
        statement="True",
        proof="by trivial",
        environment=LeanEnvironment.CORE,
    )

    assert [diagnostic.code for diagnostic in diagnostics] == [
        "LEAN_RUNTIME_SETUP_FAILED"
    ]
    assert diagnostics[0].phase is LeanDiagnosticPhase.RUNTIME_SETUP
    assert diagnostics[0].source_span is None
    assert diagnostics[0].raw_backend_message == detail
