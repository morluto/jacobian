from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.operations import OperationRequest
from jacobian.domains.logic import operations
from jacobian.domains.logic.domain_declarations import logic_operations
from jacobian.domains.logic.operations import (
    CanonicalCnf,
    CnfCanonicalizeRequest,
    LeanCheckRequest,
    SatAssignmentCheckRequest,
    SatSolveRequest,
    SmtLogic,
    SmtSolveRequest,
    canonicalize_cnf,
    check_lean_source,
    check_sat_assignment,
    solve_sat,
    solve_smt,
)
from jacobian.inline_execution import InlineOperationAdapter
from jacobian.operation_errors import OperationInvocationError
from jacobian.operations import OperationAbortError
from jacobian.process_policy import ProcessResult, ProcessTermination


def test_logic_bundle_exposes_only_atomic_inline_operations() -> None:
    assert tuple(operation.operation_id for operation in logic_operations()) == (
        "sat.cnf.canonicalize",
        "sat.assignment.check",
        "sat.solve",
        "smt.solve",
        "lean.check",
    )


def test_canonical_cnf_can_be_passed_directly_to_assignment_and_solver() -> None:
    canonical = canonicalize_cnf(
        CnfCanonicalizeRequest(
            variable_names=("b", "a"),
            clauses=((1, -2), (2,), (1, -1)),
        )
    ).cnf

    assert canonical == CanonicalCnf(variables=("a", "b"), clauses=((-1, 2), (1,)))
    assert check_sat_assignment(
        SatAssignmentCheckRequest(cnf=canonical, assignment=(True, True))
    ).satisfies
    solved = solve_sat(SatSolveRequest(cnf=canonical))
    assert solved.outcome == "SAT"
    assert solved.assignment is not None


def test_tautological_cnf_is_a_typed_invalid_request() -> None:
    with pytest.raises(ValidationError, match="non-tautological"):
        CanonicalCnf(variables=("x",), clauses=((1, -1),))

    operation = next(
        operation
        for operation in logic_operations()
        if operation.operation_id == "sat.assignment.check"
    )
    with pytest.raises(OperationInvocationError) as raised:
        InlineOperationAdapter(operation).prepare(
            OperationRequest(
                operation_id=operation.operation_id,
                input={
                    "cnf": {"variables": ["x"], "clauses": [[1, -1]]},
                    "assignment": [True],
                },
            )
        )

    assert raised.value.diagnostic.code == "INVALID_LOGIC_REQUEST"


def test_assignment_reports_the_first_unsatisfied_clause() -> None:
    result = check_sat_assignment(
        SatAssignmentCheckRequest(
            cnf=CanonicalCnf(variables=("x",), clauses=((1,),)),
            assignment=(False,),
        )
    )

    assert result.satisfies is False
    assert result.first_unsatisfied_clause == 0


def test_sat_solver_returns_unsat_without_a_model() -> None:
    result = solve_sat(
        SatSolveRequest(cnf=CanonicalCnf(variables=("x",), clauses=((-1,), (1,))))
    )

    assert result.outcome == "UNSAT"
    assert result.assignment is None


def test_smt_solver_uses_the_inline_smtlib_query() -> None:
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                "(assert (> x 0))\n"
                "(check-sat)\n"
            ),
        )
    )

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None


def test_smt_request_rejects_a_logic_name_hidden_in_a_comment() -> None:
    with pytest.raises(ValueError, match="declare the requested logic"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=("; (set-logic QF_LIA)\n(set-logic QF_UF)\n(check-sat)\n"),
        )


def test_smt_request_rejects_state_changes_after_check_sat() -> None:
    with pytest.raises(ValueError, match="end with its check-sat command"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(check-sat)\n(assert false)\n",
        )


def test_lean_check_returns_typed_rejection_without_retaining_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(operations.shutil, "which", lambda _name: "/usr/bin/lean")
    monkeypatch.setattr(
        operations,
        "execute_process",
        lambda _request: ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=1,
            stdout=b"",
            stderr=b"Snippet.lean:1:20: error: invalid proof\n",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )

    result = check_lean_source(LeanCheckRequest(source="example : True := by sorry"))

    assert result.outcome == "REJECTED"
    assert result.diagnostics == (
        operations.LeanDiagnostic(
            severity="ERROR", message="Snippet.lean:1:20: error: invalid proof"
        ),
    )


def test_lean_check_elaborates_a_bounded_inline_source() -> None:
    try:
        result = check_lean_source(
            LeanCheckRequest(source="example : True := by trivial")
        )
    except OperationAbortError as exc:
        if exc.diagnostic.code != "LEAN_UNAVAILABLE":
            raise
        pytest.skip("the fixed Lean toolchain is not installed")

    assert result.outcome == "ELABORATED"
    assert result.diagnostics == ()
