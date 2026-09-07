"""Public SMT fragment sort-admission regressions."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic._smt import SmtLogic, SmtSolveRequest, solve_smt
from jacobian.math.logic._unsat_core import (
    SmtUnsatCoreRequest,
    compute_smt_unsat_core,
)


def test_qf_lia_retains_closed_integer_arithmetic() -> None:
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(assert (= (* 7 6) 42))\n(check-sat)",
        )
    )
    assert result.outcome == "SAT"


def test_qf_lra_retains_closed_real_arithmetic() -> None:
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LRA,
            smtlib="(set-logic QF_LRA)\n(assert (= (* (/ 1.0 3.0) 3.0) 1.0))\n(check-sat)",
        )
    )
    assert result.outcome == "SAT"


def test_qf_uf_retains_boolean_uninterpreted_constants() -> None:
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_UF,
            smtlib="(set-logic QF_UF)\n(declare-const p Bool)\n(assert p)\n(check-sat)",
        )
    )
    assert result.outcome == "SAT"


@pytest.mark.parametrize("logic", (SmtLogic.QF_LIA, SmtLogic.QF_LRA, SmtLogic.QF_UF))
def test_advertised_fragments_reject_bitvector_terms(
    logic: SmtLogic,
) -> None:
    with pytest.raises(OperationDomainValidationError, match="declared"):
        solve_smt(
            SmtSolveRequest(
                logic=logic,
                smtlib=(
                    f"(set-logic {logic.value})\n"
                    "(assert (= (_ bv1 1000) (_ bv1 1000)))\n"
                    "(check-sat)"
                ),
            )
        )


def test_qf_lia_accepts_a_safe_large_closed_product() -> None:
    literal = "9" * 4_096
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)\n"
                f"(assert (= (* {literal} {literal}) (* {literal} {literal})))\n"
                "(check-sat)"
            ),
        )
    )
    assert result.outcome == "SAT"


def test_qf_lia_rejects_bounded_let_expansion_before_solving() -> None:
    expression = "9" * 100
    for _ in range(8):
        expression = f"(let ((a {expression})) (* a a))"
    with pytest.raises(OperationDomainValidationError, match=r"growth|work"):
        solve_smt(
            SmtSolveRequest(
                logic=SmtLogic.QF_LIA,
                smtlib=f"(set-logic QF_LIA)\n(assert (= {expression} 0))\n(check-sat)",
            )
        )


def test_qf_lia_rejects_repeated_closed_coefficient_scaling() -> None:
    expression = "x"
    literal = "9" * 1_000
    for index in range(32):
        expression = f"(let ((a{index} {literal})) (* a{index} {expression}))"
    with pytest.raises(OperationDomainValidationError, match=r"growth|work"):
        solve_smt(
            SmtSolveRequest(
                logic=SmtLogic.QF_LIA,
                smtlib=(
                    "(set-logic QF_LIA)\n"
                    "(declare-const x Int)\n"
                    f"(assert (= {expression} 0))\n(check-sat)"
                ),
            )
        )


@pytest.mark.parametrize(
    "assertion",
    (
        "(declare-const a (Array Int Int))\n(assert (= (select a 0) 0))",
        "(declare-const x Int)\n(declare-fun f (Int) Int)\n(assert (= (f x) x))",
    ),
)
def test_qf_lia_rejects_array_and_nonconstant_uf(assertion: str) -> None:
    with pytest.raises(OperationDomainValidationError, match="declared"):
        solve_smt(
            SmtSolveRequest(
                logic=SmtLogic.QF_LIA,
                smtlib=f"(set-logic QF_LIA)\n{assertion}\n(check-sat)",
            )
        )


def test_unsat_core_reuses_sort_and_operator_admission() -> None:
    with pytest.raises(OperationDomainValidationError, match="declared"):
        compute_smt_unsat_core(
            SmtUnsatCoreRequest(
                logic=SmtLogic.QF_LIA,
                smtlib=(
                    "(set-logic QF_LIA)\n"
                    "(declare-const x Int)\n"
                    "(declare-fun f (Int) Int)\n"
                    "(assert (= (f x) x))\n"
                    "(check-sat)"
                ),
            )
        )
