from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jacobian.math.logic import _operations as operations
from jacobian.math.logic._operations import (
    CanonicalCnf,
    CnfCanonicalizeRequest,
    LeanCheckRequest,
    LeanDeclarationKind,
    LeanDeclarationSearchRequest,
    SatAssignmentCheckRequest,
    SatSolveRequest,
    SmtLogic,
    SmtSolveRequest,
    canonicalize_cnf,
    check_lean_source,
    check_sat_assignment,
    search_mathlib_declarations,
    solve_sat,
    solve_smt,
)
from jacobian.math.logic._tools import TOOLS


def test_logic_bundle_exposes_only_atomic_inline_operations() -> None:
    assert tuple(operation.operation_id for operation in TOOLS) == (
        "sat.cnf.canonicalize",
        "sat.assignment.check",
        "sat.solve",
        "smt.solve",
        "lean.check",
        "lean.declarations.search",
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
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(operations, "_mathlib_runtime_root", lambda: tmp_path)
    monkeypatch.setattr(operations.shutil, "which", lambda _name: "/usr/bin/lake")
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


def test_lean_check_uses_the_fixed_mathlib_lake_environment(
    tmp_path, monkeypatch
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(operations, "_mathlib_runtime_root", lambda: tmp_path)
    monkeypatch.setattr(operations.shutil, "which", lambda _name: "/usr/bin/lake")

    def fake_run(arguments: list[str], **kwargs: object) -> SimpleNamespace:
        observed["arguments"] = arguments
        observed["cwd"] = kwargs["cwd"]
        return SimpleNamespace(
            returncode=0,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
            cancelled=False,
        )

    monkeypatch.setattr(operations, "run_bounded_process", fake_run)

    result = check_lean_source(
        LeanCheckRequest(source="import Mathlib\nexample : True := by trivial")
    )

    assert result.outcome == "ELABORATED"
    arguments = observed["arguments"]
    assert isinstance(arguments, list)
    assert arguments[:3] == ["/usr/bin/lake", "env", "lean"]
    assert observed["cwd"] == str(tmp_path)


def test_mathlib_search_rejects_executable_text_as_a_declaration_name() -> None:
    with pytest.raises(ValidationError, match="ASCII dotted-name grammar"):
        LeanDeclarationSearchRequest(type_constants=("Nat; #eval 1",))


def test_mathlib_search_returns_a_query_bound_display_projection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(operations, "_mathlib_runtime_root", lambda: tmp_path)
    monkeypatch.setattr(operations.shutil, "which", lambda _name: "/usr/bin/lake")
    monkeypatch.setattr(
        operations,
        "run_bounded_process",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                b'JACOBIAN_LEAN_SEARCH_RESULT {"declarations":['
                b'{"kind":"THEOREM","module":"Init.Data.Nat.Basic",'
                b'"name":"Nat.add_comm","type":"forall (n m : Nat), '
                b'n + m = m + n","type_truncated":false}],'
                b'"scanned_declarations":42,'
                b'"stop_reason":"EXHAUSTED"}\n'
            ),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
            cancelled=False,
        ),
    )
    request = LeanDeclarationSearchRequest(
        name_contains="add_comm",
        namespace_prefixes=("Nat",),
        kinds=(LeanDeclarationKind.THEOREM,),
    )

    result = search_mathlib_declarations(request)

    assert result.outcome == "COMPLETED"
    assert result.query == request
    assert result.stop_reason == "EXHAUSTED"
    assert result.declarations[0].name == "Nat.add_comm"


def test_mathlib_search_reports_an_unavailable_fixed_environment(monkeypatch) -> None:
    monkeypatch.setattr(operations, "_mathlib_runtime_root", lambda: None)

    result = search_mathlib_declarations(
        LeanDeclarationSearchRequest(name_contains="add_comm")
    )

    assert result.outcome == "UNAVAILABLE"
    assert result.declarations == ()
    assert result.stop_reason is None


def test_mathlib_runtime_rejects_a_changed_manifest(tmp_path, monkeypatch) -> None:
    manifest = (
        f'{{"packages":[{{"name":"mathlib","rev":"{operations._MATHLIB_REVISION}"}}]}}'
    ).encode()
    (tmp_path / "lake-manifest.json").write_bytes(manifest)
    (tmp_path / "lean-toolchain").write_text(
        operations._LEAN_TOOLCHAIN, encoding="utf-8"
    )
    mathlib_olean = (
        tmp_path
        / ".lake"
        / "packages"
        / "mathlib"
        / ".lake"
        / "build"
        / "lib"
        / "lean"
        / "Mathlib.olean"
    )
    mathlib_olean.parent.mkdir(parents=True)
    mathlib_olean.touch()
    monkeypatch.setenv("JACOBIAN_MATHLIB_ROOT", str(tmp_path))
    monkeypatch.setattr(
        operations,
        "_MATHLIB_MANIFEST_SHA256",
        operations.hashlib.sha256(manifest).hexdigest(),
    )

    assert operations._mathlib_runtime_root() == tmp_path

    (tmp_path / "lake-manifest.json").write_bytes(manifest + b"\n")

    assert operations._mathlib_runtime_root() is None
