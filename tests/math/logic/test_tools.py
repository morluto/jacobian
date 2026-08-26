from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jacobian.math.logic import _sat as sat
from jacobian.math.logic import _smt as smt
from jacobian.math.logic._cnf import (
    CanonicalCnf,
    CnfCanonicalizeRequest,
    SatAssignmentCheckRequest,
    canonicalize_cnf,
    check_sat_assignment,
)
from jacobian.math.logic._sat import (
    LprAddition,
    LprDeletion,
    LprPropagationHint,
    SatLprRefutation,
    SatRefutationCheckRequest,
    SatSolveRequest,
    SatSolveResult,
    check_sat_refutation,
    solve_sat,
)
from jacobian.math.logic._smt import (
    SmtLogic,
    SmtSolveRequest,
    SmtSolveResult,
    solve_smt,
)
from jacobian.math.logic._tools import TOOLS
from jacobian.math.logic._unsat_core import SmtUnsatCoreRequest


@contextmanager
def raises_logic_validation():
    with pytest.raises(ValidationError) as error:
        yield error
    assert error.value.errors()[0]["type"].startswith("logic.")


def test_logic_bundle_exposes_only_atomic_inline_operations() -> None:
    assert tuple(operation.operation_id for operation in TOOLS) == (
        "sat.cnf.canonicalize",
        "sat.assignment.check",
        "sat.solve",
        "sat.refutation.check",
        "smt.solve",
        "smt.unsat_core",
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
    with raises_logic_validation():
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

    assert sat._dimacs_cnf(request.cnf) == b"p cnf 1 2\n-1 0\n1 0\n"
    assert sat._ascii_lpr(request.refutation) == b"3 0 1 2 0\n"


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

    assert sat._ascii_lpr(refutation) == b"0 d 1 0\n3 1 1 -1 0 2 -2 0\n"


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
        sat._ascii_lpr(request.refutation)
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
    monkeypatch.setattr(sat, "_MAX_LPR_RESULT_BYTES", 1)

    with pytest.raises(ValueError, match="source-bound result limit"):
        _unit_refutation_request()


def test_lpr_refutation_returns_unavailable_without_the_pinned_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sat.shutil, "which", lambda _name: None)

    result = check_sat_refutation(_unit_refutation_request())

    assert result.outcome == "UNAVAILABLE"
    assert result.cnf == _unit_refutation_request().cnf
    assert result.refutation == _unit_refutation_request().refutation


def test_lpr_refutation_accepts_only_the_exact_cake_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sat.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sat, "_cake_lpr_is_supported", lambda _path: True)
    monkeypatch.setattr(
        sat,
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
    monkeypatch.setattr(sat.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sat, "_cake_lpr_is_supported", lambda _path: True)
    monkeypatch.setattr(
        sat,
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
    monkeypatch.setattr(sat.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sat, "_cake_lpr_is_supported", lambda _path: True)
    monkeypatch.setattr(sat, "run_bounded_process", lambda *_args, **_kwargs: completed)

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


def _left_nested_additions(levels: int) -> str:
    expression = "0"
    for _ in range(levels):
        expression = f"(+ {expression} 0)"
    return expression


def _pigeonhole_smtlib(pigeons: int, holes: int) -> str:
    lines = ["(set-logic QF_LIA)"]
    for pigeon in range(pigeons):
        for hole in range(holes):
            lines.append(f"(declare-const b_{pigeon}_{hole} Int)")
    for pigeon in range(pigeons):
        row = " ".join(f"b_{pigeon}_{hole}" for hole in range(holes))
        lines.append(f"(assert (= (+ {row}) 1))")
    for hole in range(holes):
        column = " ".join(f"b_{pigeon}_{hole}" for pigeon in range(pigeons))
        lines.append(f"(assert (<= (+ {column}) 1))")
    for pigeon in range(pigeons):
        for hole in range(holes):
            lines.append(
                f"(assert (and (>= b_{pigeon}_{hole} 0) (<= b_{pigeon}_{hole} 1)))"
            )
    lines.append("(check-sat)")
    return "\n".join(lines)


def test_smt_request_rejects_nesting_beyond_the_term_depth_budget() -> None:
    with pytest.raises(ValueError, match="maximum term depth"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (> x {_left_nested_additions(513)}))\n"
                "(check-sat)\n"
            ),
        )


def test_smt_request_admits_nesting_at_the_term_depth_boundary_and_still_solves() -> (
    None
):
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (> x {_left_nested_additions(509)}))\n"
                "(check-sat)\n"
            ),
        )
    )

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None


def test_smt_request_rejects_more_than_the_compound_term_budget() -> None:
    block = "(" * 400 + ")" * 400
    with pytest.raises(ValueError, match="compound terms"):
        SmtSolveRequest(logic=SmtLogic.QF_UF, smtlib=block * 100)


def test_smt_request_admits_a_large_shallow_formula_and_still_solves() -> None:
    assertions = "\n".join(
        ["(set-logic QF_LIA)", "(declare-const x Int)"]
        + ["(assert (= x (+ 1 1)))"] * 5_000
    )
    result = solve_smt(
        SmtSolveRequest(logic=SmtLogic.QF_LIA, smtlib=assertions + "\n(check-sat)")
    )

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None


def test_smt_request_rejects_a_numeral_wider_than_the_digit_budget() -> None:
    with pytest.raises(ValueError, match="numeral wider"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (= x {'9' * 4_097}))\n"
                "(check-sat)\n"
            ),
        )
    with pytest.raises(ValueError, match="numeral wider"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (= x {'9' * 100_000}))\n"
                "(check-sat)\n"
            ),
        )


def test_smt_request_admits_a_numeral_at_the_digit_boundary_and_still_solves() -> None:
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (= x {'9' * 4_096}))\n"
                "(check-sat)\n"
            ),
        )
    )

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None


def test_smt_request_admits_a_literal_beyond_the_former_ceiling_with_its_exact_model() -> (
    None
):
    literal = "3" + "1" * 191 + "7"
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (= x {literal}))\n"
                "(check-sat)\n"
            ),
        )
    )

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None
    assert literal in result.model_smtlib


def test_atom_numeral_weight_classifies_literal_tokens_not_symbols() -> None:
    assert smt._atom_numeral_weight("9" * 193) == 193
    assert smt._atom_numeral_weight("007") == 3
    assert smt._atom_numeral_weight("1234.5678") == 8
    assert smt._atom_numeral_weight("1.") == 1
    assert smt._atom_numeral_weight("#xdeadbeef") == 8
    assert smt._atom_numeral_weight("#b1010") == 4
    assert smt._atom_numeral_weight("v" + "0" * 193) == 0
    assert smt._atom_numeral_weight("a1.b2") == 0
    assert smt._atom_numeral_weight(":named") == 0
    assert smt._atom_numeral_weight("1.2.3") == 0


def test_smt_request_admits_digits_inside_simple_symbols_and_still_solves() -> None:
    symbol = "v" + "0" * 193
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                f"(declare-const {symbol} Int)\n"
                f"(assert (> {symbol} 0))\n"
                "(check-sat)\n"
            ),
        )
    )

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None


def test_smt_request_rejects_a_decimal_wider_than_the_digit_budget() -> None:
    for numeral in ("9" * 4_097 + ".5", "0." + "9" * 4_097):
        with pytest.raises(ValueError, match="numeral wider"):
            SmtSolveRequest(
                logic=SmtLogic.QF_LRA,
                smtlib=(
                    "(set-logic QF_LRA)\n"
                    "(declare-const x Real)\n"
                    f"(assert (= x {numeral}))\n"
                    "(check-sat)\n"
                ),
            )


def test_smt_request_admits_a_decimal_at_the_digit_boundary_and_still_solves() -> None:
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LRA,
            smtlib=(
                "(set-logic QF_LRA)\n"
                "(declare-const x Real)\n"
                f"(assert (= x 99.{'9' * 4_094}))\n"
                "(check-sat)\n"
            ),
        )
    )

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None


def test_smtlib_structure_weights_indexed_bit_vector_values_not_bv_symbols() -> None:
    assert smt._smtlib_structure(f"(_ bv{'9' * 193} 8)").numeral_digits == 193
    assert smt._smtlib_structure("(_ bv1010 8)").numeral_digits == 4
    assert smt._smtlib_structure("(declare-const bv1010 Int)").numeral_digits == 0
    assert (
        smt._smtlib_structure("(assert (= x |bv" + "9" * 193 + "|))").numeral_digits
        == 0
    )


def test_smt_request_rejects_an_indexed_bit_vector_value_beyond_the_digit_budget() -> (
    None
):
    with pytest.raises(ValueError, match="numeral wider"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (= x (_ bv{'9' * 4_097} 8)))\n"
                "(check-sat)\n"
            ),
        )


def test_smt_request_rejects_an_ill_sorted_indexed_bit_vector_equality() -> None:
    """Lexical digit admission cannot override backend well-sortedness."""

    with pytest.raises(ValidationError) as error:
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (= x (_ bv{'9' * 4_096} 8)))\n"
                "(check-sat)\n"
            ),
        )

    assert error.value.errors(include_url=False)[0]["type"] == "logic.smtlib_grammar"


def test_smt_request_admits_a_bv_named_symbol_outside_index_context_and_solves() -> (
    None
):
    symbol = "bv" + "0" * 193
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                f"(declare-const {symbol} Int)\n"
                f"(assert (> {symbol} 0))\n"
                "(check-sat)\n"
            ),
        )
    )

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None


def test_smt_request_schema_publishes_the_structural_limits() -> None:
    schema = SmtSolveRequest.model_json_schema()
    description = schema["properties"]["smtlib"]["description"]

    assert f"nesting depth at most {smt._MAX_SMTLIB_DEPTH}" in description
    assert f"compound terms at most {smt._MAX_SMTLIB_TERMS}" in description
    assert f"declared symbols at most {smt._MAX_SMTLIB_DECLARATIONS}" in description
    assert f"at most {smt._MAX_SMTLIB_NUMERAL_DIGITS} digits" in description


def test_smt_request_rejects_more_than_the_declaration_budget() -> None:
    declarations = "\n".join(f"(declare-const v{index} Int)" for index in range(4_097))
    with pytest.raises(ValueError, match="declares more than"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=f"(set-logic QF_LIA)\n{declarations}\n(check-sat)",
        )


def test_smt_request_admits_at_the_declaration_boundary_and_still_solves() -> None:
    declarations = "\n".join(f"(declare-const v{index} Int)" for index in range(4_096))
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                f"(set-logic QF_LIA)\n{declarations}\n(assert (= v0 1))\n(check-sat)"
            ),
        )
    )

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None


def test_structural_rejection_precedes_z3_parsing(monkeypatch) -> None:
    import z3

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Z3 parsed a request that admission had to reject")

    monkeypatch.setattr(z3, "parse_smt2_string", refuse)
    with pytest.raises(ValueError, match="numeral wider"):
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (= x {'9' * 4_097}))\n"
                "(check-sat)\n"
            ),
        )


def test_smt_solver_passes_time_work_and_memory_budgets_to_z3(monkeypatch) -> None:
    import z3

    recorded: dict[str, int] = {}

    class RecordingSolver:
        def set(self, **settings: int) -> None:
            recorded.update(settings)

        def add(self, _assertions: object) -> None:
            return None

        def check(self) -> object:
            return z3.unsat

    monkeypatch.setattr(z3, "SolverFor", lambda _logic: RecordingSolver())
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
            timeout_ms=2_500,
        )
    )

    assert result.outcome == "UNSAT"
    assert recorded == {
        "timeout": 2_500,
        "rlimit": smt._SOLVER_RLIMIT,
        "max_memory": smt._SOLVER_MAX_MEMORY_MB,
    }


def test_smt_solver_projects_exhausted_work_budget_as_typed_unknown(
    monkeypatch,
) -> None:
    monkeypatch.setattr(smt, "_SOLVER_RLIMIT", 1)
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        )
    )

    assert result.outcome == "UNKNOWN"
    assert result.exhausted == "work"
    assert result.model_smtlib is None
    assert result.detail is not None and "work budget" in result.detail


def test_sat_solver_projects_exhausted_work_budget_as_typed_unknown(
    monkeypatch,
) -> None:
    monkeypatch.setattr(smt, "_SOLVER_RLIMIT", 1)
    result = solve_sat(
        SatSolveRequest(cnf=CanonicalCnf(variables=("x",), clauses=((1,),)))
    )

    assert result.outcome == "UNKNOWN"
    assert result.exhausted == "work"
    assert result.assignment is None
    assert result.detail is not None and "work budget" in result.detail


def test_smt_solver_projects_exhausted_memory_budget_as_typed_unknown(
    monkeypatch,
) -> None:
    monkeypatch.setattr(smt, "_SOLVER_MAX_MEMORY_MB", 2)
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=_pigeonhole_smtlib(8, 7),
        )
    )

    assert result.outcome == "UNKNOWN"
    assert result.exhausted == "memory"
    assert result.detail is not None and "memory budget" in result.detail


def test_smt_solver_projects_exhausted_time_budget_as_typed_unknown() -> None:
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=_pigeonhole_smtlib(8, 7),
            timeout_ms=1,
        )
    )

    assert result.outcome == "UNKNOWN"
    assert result.exhausted == "time"
    assert result.detail is not None and "time budget" in result.detail


def test_smt_solver_wraps_backend_failure_during_the_solve(monkeypatch) -> None:
    import z3

    class ExplodingSolver:
        def set(self, **_settings: int) -> None:
            return None

        def add(self, _assertions: object) -> None:
            return None

        def check(self) -> object:
            raise z3.Z3Exception("backend exploded")

    monkeypatch.setattr(z3, "SolverFor", lambda _logic: ExplodingSolver())
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        )
    )

    assert result.outcome == "UNKNOWN"
    assert result.exhausted is None
    assert result.detail is not None and "backend exploded" in result.detail


@pytest.mark.parametrize(
    ("message", "exhausted"),
    (
        ("out of memory", "memory"),
        ("max. memory limit exceeded", "memory"),
        ("max. resource limit exceeded", "work"),
        ("canceled during solve", "work"),
        ("exceeded the time limit before a verdict", "time"),
    ),
)
def test_sat_solver_projects_exhaustion_messages_from_z3_exceptions(
    monkeypatch,
    message: str,
    exhausted: str,
) -> None:
    import z3

    class ExhaustingSolver:
        def set(self, **_settings: int) -> None:
            return None

        def add(self, _assertions: object) -> None:
            raise z3.Z3Exception(message)

        def check(self) -> object:
            raise AssertionError("check must not run after a failed assertion add")

    monkeypatch.setattr(z3, "Solver", ExhaustingSolver)
    result = solve_sat(
        SatSolveRequest(cnf=CanonicalCnf(variables=("x",), clauses=((1,),)))
    )

    assert result.outcome == "UNKNOWN"
    assert result.exhausted == exhausted
    assert result.assignment is None
    assert result.detail == smt._EXHAUSTION_DETAILS[exhausted]


@pytest.mark.parametrize(
    ("message", "exhausted"),
    (
        ("out of memory", "memory"),
        ("max. resource limit exceeded", "work"),
        ("timeout exceeded while checking", "time"),
    ),
)
def test_smt_solver_projects_exhaustion_messages_from_z3_exceptions(
    monkeypatch,
    message: str,
    exhausted: str,
) -> None:
    import z3

    class ExhaustingSolver:
        def set(self, **_settings: int) -> None:
            return None

        def add(self, _assertions: object) -> None:
            raise z3.Z3Exception(message)

        def check(self) -> object:
            raise AssertionError("check must not run after a failed assertion add")

    monkeypatch.setattr(z3, "SolverFor", lambda _logic: ExhaustingSolver())
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        )
    )

    assert result.outcome == "UNKNOWN"
    assert result.exhausted == exhausted
    assert result.model_smtlib is None
    assert result.detail == smt._EXHAUSTION_DETAILS[exhausted]


def test_smt_solver_projects_exhaustion_from_model_serialization(monkeypatch) -> None:
    import z3

    class ExhaustingModel:
        def sexpr(self) -> str:
            raise z3.Z3Exception("out of memory")

    class SolvingThenExhaustingSolver:
        def set(self, **_settings: int) -> None:
            return None

        def add(self, _assertions: object) -> None:
            return None

        def check(self) -> object:
            return z3.sat

        def model(self) -> object:
            return ExhaustingModel()

    monkeypatch.setattr(z3, "SolverFor", lambda _logic: SolvingThenExhaustingSolver())
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        )
    )

    assert result.outcome == "UNKNOWN"
    assert result.exhausted == "memory"
    assert result.detail == "the bounded solver memory budget was exhausted"


def test_solver_wraps_unrecognized_z3_exceptions_without_typed_exhaustion(
    monkeypatch,
) -> None:
    import z3

    class ExplodingSolver:
        def set(self, **_settings: int) -> None:
            return None

        def add(self, _assertions: object) -> None:
            raise z3.Z3Exception("backend exploded")

        def check(self) -> object:
            raise AssertionError("check must not run after a failed assertion add")

    monkeypatch.setattr(z3, "Solver", ExplodingSolver)
    sat_result = solve_sat(
        SatSolveRequest(cnf=CanonicalCnf(variables=("x",), clauses=((1,),)))
    )
    monkeypatch.setattr(z3, "SolverFor", lambda _logic: ExplodingSolver())
    smt_result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        )
    )

    for result in (sat_result, smt_result):
        assert result.outcome == "UNKNOWN"
        assert result.exhausted is None
        assert result.detail is not None
        assert (
            "the Z3 backend failed during the bounded solve: backend exploded"
            in result.detail
        )


@pytest.mark.parametrize(
    ("operation", "accepted_request"),
    (
        (
            solve_sat,
            SatSolveRequest(
                cnf=CanonicalCnf(variables=("x",), clauses=((1,),)),
            ),
        ),
        (
            solve_smt,
            SmtSolveRequest(
                logic=SmtLogic.QF_LIA,
                smtlib=(
                    "(set-logic QF_LIA)\n"
                    "(declare-const x Int)\n"
                    "(assert (> x 0))\n"
                    "(check-sat)"
                ),
            ),
        ),
    ),
)
def test_z3_initialization_failure_is_a_typed_unknown(
    operation,
    accepted_request,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Z3 module initialization cannot escape an accepted request."""

    monkeypatch.setitem(sys.modules, "z3", None)

    result = operation(accepted_request)

    assert result.outcome == "UNKNOWN"
    assert result.exhausted is None
    assert result.detail is not None
    assert "could not initialize" in result.detail
    assert type(result).model_validate(result.model_dump()) == result


def test_smt_request_admission_skips_grammar_rejection_without_the_backend(
    monkeypatch,
) -> None:
    """Backend absence at admission stays the typed execution init outcome."""

    monkeypatch.setitem(sys.modules, "z3", None)
    admitted = SmtSolveRequest(
        logic=SmtLogic.QF_LIA,
        smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
    )

    result = solve_smt(admitted)

    assert result.outcome == "UNKNOWN"
    assert result.exhausted is None
    assert result.detail is not None
    assert "could not initialize" in result.detail


def test_smt_request_rejects_malformed_source_as_a_request_error(monkeypatch) -> None:
    """Malformed SMT-LIB stays caller-correctable instead of solver indeterminacy."""

    import z3

    def refuse_solver(_logic: object) -> object:
        raise AssertionError("a malformed request reached solver construction")

    monkeypatch.setattr(z3, "SolverFor", refuse_solver)
    with pytest.raises(ValidationError) as error:
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                "(assert (> y 0))\n"
                "(check-sat)"
            ),
        )

    assert error.value.errors(include_url=False)[0]["type"] == "logic.smtlib_grammar"


@pytest.mark.parametrize(
    "identifier",
    ("memory", "timeout", "canceled", "|resource limit|"),
)
def test_smt_request_rejects_undeclared_identifiers_named_after_resource_keywords(
    identifier: str,
) -> None:
    """A located diagnostic quoting a caller symbol is never budget exhaustion.

    Z3 reports an undeclared identifier as
    ``(error "line L column C: unknown constant <identifier>")``; when the
    caller-controlled spelling contains a resource keyword, that text must
    still be a caller-correctable grammar rejection rather than an admitted
    request whose execution reports an exhausted budget.
    """

    with pytest.raises(ValidationError) as error:
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                f"(assert (> {identifier} 0))\n"
                "(check-sat)"
            ),
        )

    assert error.value.errors(include_url=False)[0]["type"] == "logic.smtlib_grammar"


def test_smt_solver_types_parse_stage_backend_failures_as_unknown(monkeypatch) -> None:
    """A backend parser failure on an admitted source is execution UNKNOWN."""

    import z3

    def exhausting_parser(_source: str) -> object:
        raise z3.Z3Exception("parser ran out of memory")

    admitted = SmtSolveRequest(
        logic=SmtLogic.QF_LIA,
        smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
    )
    monkeypatch.setattr(z3, "parse_smt2_string", exhausting_parser)
    result = solve_smt(admitted)

    assert result.outcome == "UNKNOWN"
    assert result.exhausted == "memory"
    assert result.detail == smt._EXHAUSTION_DETAILS["memory"]


def test_smt_solver_never_classifies_located_source_text_as_exhaustion(
    monkeypatch,
) -> None:
    """Located diagnostic text quotes the caller's source, not a budget.

    A located ``(error "line ... column ...: ...")`` diagnostic embeds
    caller-controlled spellings, so its resource keywords cannot establish an
    exhausted budget; execution reports it as a generic backend failure.
    """

    import z3

    def diagnosing_parser(_source: str) -> object:
        raise z3.Z3Exception(b'(error "line 1 column 0: out of memory")')

    admitted = SmtSolveRequest(
        logic=SmtLogic.QF_LIA,
        smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
    )
    monkeypatch.setattr(z3, "parse_smt2_string", diagnosing_parser)

    result = solve_smt(admitted)

    assert result.outcome == "UNKNOWN"
    assert result.exhausted is None
    assert result.detail is not None and "bounded solve" in result.detail


def test_smt_request_admission_rejects_located_parser_diagnostics(monkeypatch) -> None:
    """A located Z3 parser diagnostic stays a caller-correctable rejection."""

    import z3

    def diagnosing_parser(_source: str) -> object:
        raise z3.Z3Exception(b'(error "line 2 column 11: unknown constant y")\n')

    monkeypatch.setattr(z3, "parse_smt2_string", diagnosing_parser)
    with pytest.raises(ValidationError) as error:
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                "(assert (> y 0))\n"
                "(check-sat)"
            ),
        )

    assert error.value.errors(include_url=False)[0]["type"] == "logic.smtlib_grammar"


def test_smt_request_admission_defers_parse_stage_os_errors(monkeypatch) -> None:
    """A parse-stage native backend failure stays off the caller's hands.

    ``math.run`` validates the request before calling the solver, so admission
    must not let a native ``OSError`` from ``parse_smt2_string`` escape as a
    host exception; it carries no evidence about the source and defers to
    execution, which reports it through the typed UNKNOWN translation.
    """

    import z3

    def failing_parser(_source: str) -> object:
        raise OSError("native parser backend unavailable")

    source = "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)"
    monkeypatch.setattr(z3, "parse_smt2_string", failing_parser)
    admitted = SmtSolveRequest(logic=SmtLogic.QF_LIA, smtlib=source)
    result = solve_smt(admitted)

    assert result.outcome == "UNKNOWN"


@pytest.mark.parametrize(
    "request_type",
    sorted(
        {SmtSolveRequest, SmtUnsatCoreRequest, *SmtSolveRequest.__subclasses__()},
        key=lambda request_type: request_type.__name__,
    ),
    ids=lambda request_type: request_type.__name__,
)
def test_every_concrete_smt_request_rejects_malformed_grammar(request_type) -> None:
    """No SMT request subclass may leave malformed grammar to execution."""

    with pytest.raises(ValidationError):
        request_type(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                "(assert (> y 0))\n"
                "(check-sat)"
            ),
        )


def test_unknown_projection_maps_every_exhausted_resource() -> None:
    assert smt._project_unknown("max. resource limit exceeded") == (
        "work",
        "the bounded solver work budget was exhausted",
    )
    assert smt._project_unknown("canceled")[0] == "work"
    assert smt._project_unknown("max. memory exceeded")[0] == "memory"
    assert smt._project_unknown("timeout")[0] == "time"
    passthrough = smt._project_unknown("max. engine depth reached")
    assert passthrough[0] is None
    assert passthrough[1] == "max. engine depth reached"
    assert smt._project_unknown(None)[1].endswith("no completeness evidence")


def test_result_models_bind_exhausted_budgets_to_unknown_outcomes() -> None:
    with pytest.raises(ValueError, match="exhausted budget"):
        SatSolveResult(outcome="SAT", assignment=(True,), exhausted="time")
    with pytest.raises(ValueError, match="exhausted budget"):
        SmtSolveResult(outcome="UNSAT", exhausted="memory")
