from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jacobian.math.logic import _operations as operations
from jacobian.math.logic._operations import (
    CanonicalCnf,
    CnfCanonicalizeRequest,
    LeanCheckRequest,
    LprAddition,
    LprDeletion,
    LprPropagationHint,
    SatAssignmentCheckRequest,
    SatLprRefutation,
    SatRefutationCheckRequest,
    SatSolveRequest,
    SmtLogic,
    SmtSolveRequest,
    canonicalize_cnf,
    check_lean_source,
    check_sat_assignment,
    check_sat_refutation,
    solve_sat,
    solve_smt,
)
from jacobian.math.logic._tools import TOOLS


def test_logic_bundle_exposes_only_atomic_inline_operations() -> None:
    assert tuple(operation.operation_id for operation in TOOLS) == (
        "sat.cnf.canonicalize",
        "sat.assignment.check",
        "sat.solve",
        "sat.refutation.check",
        "smt.solve",
        "lean.check",
    )


def test_lpr_schema_explains_source_binding_and_live_clause_rules() -> None:
    operation = next(
        item for item in TOOLS if item.operation_id == "sat.refutation.check"
    )
    schema = operation.request_type.model_json_schema()
    text = str(schema)

    assert "currently live clause IDs" in text
    assert "exact one-based canonical clause order" in text
    assert "literal-inspection work" in text


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


def _unit_refutation_request() -> SatRefutationCheckRequest:
    return SatRefutationCheckRequest(
        cnf=CanonicalCnf(variables=("x",), clauses=((-1,), (1,))),
        refutation=SatLprRefutation(
            steps=(
                LprAddition(
                    clause_id=3,
                    clause=(),
                    at_hint_clause_ids=(1, 2),
                    propagation_hints=(),
                ),
            )
        ),
    )


def test_lpr_refutation_serializes_the_typed_profile_without_backend_syntax() -> None:
    request = _unit_refutation_request()

    assert operations._dimacs_cnf(request.cnf) == b"p cnf 1 2\n-1 0\n1 0\n"
    assert operations._ascii_lpr(request.refutation) == b"3 0 1 2 0\n"


def test_lpr_refutation_keeps_witnesses_and_deletions_structural() -> None:
    refutation = SatLprRefutation(
        steps=(
            LprDeletion(kind="deletion", clause_ids=(1,)),
            LprAddition(
                clause_id=3,
                clause=(1,),
                witness=(1, -1),
                at_hint_clause_ids=(2,),
                propagation_hints=(
                    LprPropagationHint(clause_id=2, at_hint_clause_ids=()),
                ),
            ),
        )
    )

    assert operations._ascii_lpr(refutation) == b"0 d 1 0\n3 1 1 -1 0 2 -2 0\n"


def test_lpr_refutation_rejects_out_of_axis_literals_and_non_live_hints() -> None:
    with pytest.raises(ValueError, match="outside the CNF variable axis"):
        SatRefutationCheckRequest(
            cnf=CanonicalCnf(variables=("x",), clauses=((1,),)),
            refutation=SatLprRefutation(
                steps=(
                    LprAddition(
                        clause_id=2,
                        clause=(2,),
                        at_hint_clause_ids=(1,),
                        propagation_hints=(),
                    ),
                )
            ),
        )
    with pytest.raises(ValueError, match="non-live clause ID 2"):
        SatRefutationCheckRequest(
            cnf=CanonicalCnf(variables=("x",), clauses=((1,),)),
            refutation=SatLprRefutation(
                steps=(
                    LprAddition(
                        clause_id=2,
                        clause=(),
                        at_hint_clause_ids=(2,),
                        propagation_hints=(),
                    ),
                )
            ),
        )


def test_lpr_refutation_admits_sparse_solver_assigned_clause_labels() -> None:
    request = SatRefutationCheckRequest(
        cnf=CanonicalCnf(variables=("x", "y"), clauses=((-1,), (1,), (-2,), (2,))),
        refutation=SatLprRefutation(
            steps=(
                LprAddition(
                    clause_id=50_000,
                    clause=(1,),
                    at_hint_clause_ids=(2,),
                    propagation_hints=(),
                ),
                LprAddition(
                    clause_id=4_000_000_000,
                    clause=(),
                    at_hint_clause_ids=(1, 2),
                    propagation_hints=(
                        LprPropagationHint(clause_id=50_000, at_hint_clause_ids=()),
                    ),
                ),
                LprDeletion(kind="deletion", clause_ids=(50_000,)),
            )
        ),
    )

    assert (
        operations._ascii_lpr(request.refutation)
        == b"50000 1 0 2 0\n4000000000 0 1 2 -50000 0\n0 d 50000 0\n"
    )


def test_lpr_refutation_still_binds_sparse_labels_to_live_clauses() -> None:
    cnf = CanonicalCnf(variables=("x",), clauses=((-1,), (1,)))
    with pytest.raises(ValueError, match="may not overwrite a live clause ID"):
        SatRefutationCheckRequest(
            cnf=cnf,
            refutation=SatLprRefutation(
                steps=(
                    LprAddition(
                        clause_id=50_000,
                        clause=(1,),
                        at_hint_clause_ids=(1,),
                        propagation_hints=(),
                    ),
                    LprAddition(
                        clause_id=50_000,
                        clause=(1,),
                        at_hint_clause_ids=(1,),
                        propagation_hints=(),
                    ),
                )
            ),
        )
    with pytest.raises(ValueError, match="non-live clause ID"):
        SatRefutationCheckRequest(
            cnf=cnf,
            refutation=SatLprRefutation(
                steps=(
                    LprAddition(
                        clause_id=50_000,
                        clause=(),
                        at_hint_clause_ids=(1, 2),
                        propagation_hints=(
                            LprPropagationHint(clause_id=60_000, at_hint_clause_ids=()),
                        ),
                    ),
                )
            ),
        )
    with pytest.raises(ValueError, match="non-live clause ID"):
        SatRefutationCheckRequest(
            cnf=cnf,
            refutation=SatLprRefutation(
                steps=(LprDeletion(kind="deletion", clause_ids=(50_000,)),)
            ),
        )


def test_lpr_refutation_reserves_the_transport_result_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operations, "_MAX_LPR_RESULT_BYTES", 1)

    with pytest.raises(ValueError, match="source-bound result limit"):
        _unit_refutation_request()


def test_lpr_refutation_returns_unavailable_without_the_pinned_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operations.shutil, "which", lambda _name: None)

    result = check_sat_refutation(_unit_refutation_request())

    assert result.outcome == "UNAVAILABLE"
    assert result.cnf == _unit_refutation_request().cnf
    assert result.refutation == _unit_refutation_request().refutation


def test_lpr_refutation_accepts_only_the_exact_cake_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operations.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(operations, "_cake_lpr_is_supported", lambda _path: True)
    monkeypatch.setattr(
        operations,
        "run_bounded_process",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"s VERIFIED UNSAT\n",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
            cancelled=False,
        ),
    )

    result = check_sat_refutation(_unit_refutation_request())

    assert result.outcome == "VALID_REFUTATION"
    assert result.detail is None


def test_lpr_refutation_projects_a_recognized_checker_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operations.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(operations, "_cake_lpr_is_supported", lambda _path: True)
    monkeypatch.setattr(
        operations,
        "run_bounded_process",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"",
            stderr=b"c empty clause not derived at end of proof\n",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
            cancelled=False,
        ),
    )

    result = check_sat_refutation(
        SatRefutationCheckRequest(
            cnf=CanonicalCnf(variables=("x",), clauses=((-1,), (1,))),
            refutation=SatLprRefutation(steps=()),
        )
    )

    assert result.outcome == "INVALID_REFUTATION"
    assert "does not derive contradiction" in (result.detail or "")


@pytest.mark.parametrize(
    ("completed", "outcome"),
    (
        (
            SimpleNamespace(
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=True,
                cancelled=False,
            ),
            "TIMEOUT",
        ),
        (
            SimpleNamespace(
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=False,
                cancelled=True,
            ),
            "ERROR",
        ),
        (
            SimpleNamespace(
                returncode=0,
                stdout=b"s VERIFIED UNSAT\ntrailing\n",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=False,
                cancelled=False,
            ),
            "ERROR",
        ),
    ),
)
def test_lpr_refutation_never_upgrades_process_failure_to_a_math_verdict(
    monkeypatch: pytest.MonkeyPatch,
    completed: SimpleNamespace,
    outcome: str,
) -> None:
    monkeypatch.setattr(operations.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(operations, "_cake_lpr_is_supported", lambda _path: True)
    monkeypatch.setattr(
        operations, "run_bounded_process", lambda *_args, **_kwargs: completed
    )

    result = check_sat_refutation(_unit_refutation_request())

    assert result.outcome == outcome
    assert result.cnf == _unit_refutation_request().cnf
    assert result.refutation == _unit_refutation_request().refutation


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


def test_smt_request_rejects_multiple_check_sat_commands() -> None:
    with pytest.raises(ValueError, match="exactly one check-sat"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(check-sat)\n(check-sat)\n",
        )


def test_smt_request_rejects_non_ascii_input() -> None:
    with pytest.raises(ValueError, match="must be ASCII"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n\xe9\n(check-sat\n",
        )


def test_lean_check_returns_typed_rejection_without_retaining_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(operations.shutil, "which", lambda _name: "/usr/bin/lean")
    monkeypatch.setattr(
        operations,
        "run_bounded_process",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"Snippet.lean:1:20: error: invalid proof\n",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
            cancelled=False,
        ),
    )

    result = check_lean_source(LeanCheckRequest(source="example : True := by sorry"))

    assert result.outcome == "REJECTED"
    assert result.diagnostics == (
        operations.LeanDiagnostic(
            severity="ERROR", message="Snippet.lean:1:20: error: invalid proof"
        ),
    )
