"""Bounded QF_BV queries in smt.solve (#2274)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic._smt import (
    SmtLogic,
    SmtSolveRequest,
    SmtSolveResult,
    solve_smt,
)


def _solve(body: str) -> SmtSolveResult:
    return solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_BV, smtlib=f"(set-logic QF_BV)\n{body}\n(check-sat)"
        )
    )


def test_issue_example_is_sat_with_exact_model() -> None:
    result = _solve("(declare-const x (_ BitVec 8))\n(assert (= (bvadd x #x01) #x00))")

    assert result.outcome == "SAT"
    assert result.model_smtlib is not None
    assert "#xff" in result.model_smtlib


def test_model_value_is_exact_by_unsat_pair() -> None:
    # x + 1 == 0 is SAT, and x != 255 is UNSAT under it: x is exactly 255.
    assert (
        _solve(
            "(declare-const x (_ BitVec 8))\n"
            "(assert (= (bvadd x #x01) #x00))\n"
            "(assert (not (= x #xff)))"
        ).outcome
        == "UNSAT"
    )


def test_modular_wraparound() -> None:
    assert _solve("(assert (= (bvadd #xff #x01) #x00))").outcome == "SAT"


def test_strict_self_comparison_is_unsat() -> None:
    assert (
        _solve(
            "(declare-const x (_ BitVec 8))\n(assert (bvult x x))",
        ).outcome
        == "UNSAT"
    )


def test_signedness_is_preserved() -> None:
    # 0xff is -1 signed (less than zero) but 255 unsigned (not less).
    assert (
        _solve(
            "(declare-const x (_ BitVec 8))\n"
            "(assert (= x #xff))\n"
            "(assert (bvslt x #x00))"
        ).outcome
        == "SAT"
    )
    assert (
        _solve(
            "(declare-const x (_ BitVec 8))\n"
            "(assert (= x #xff))\n"
            "(assert (bvult x #x00))"
        ).outcome
        == "UNSAT"
    )


def test_extract_concat_round_trip() -> None:
    assert (
        _solve(
            "(declare-const x (_ BitVec 8))\n"
            "(assert (= (concat ((_ extract 7 4) x) ((_ extract 3 0) x)) x))"
        ).outcome
        == "SAT"
    )


def test_shifts_and_bitwise_logic() -> None:
    assert (
        _solve(
            "(declare-const x (_ BitVec 8))\n"
            "(assert (= (bvshl x #x01) #x02))\n"
            "(assert (= (bvand x #x0f) #x01))"
        ).outcome
        == "SAT"
    )


def test_integer_sorted_terms_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="declared"):
        _solve("(declare-const x Int)\n(assert (> x 0))")


def test_uninterpreted_functions_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="declared"):
        _solve(
            "(declare-fun f ((_ BitVec 8)) (_ BitVec 8))\n"
            "(declare-const x (_ BitVec 8))\n"
            "(assert (= (f x) #x00))"
        )


def test_conversions_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="declared"):
        _solve("(declare-const x (_ BitVec 8))\n(assert (= (bv2int x) 0))")


def test_float_sorted_terms_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="declared"):
        _solve("(declare-const f (_ FloatingPoint 8 24))\n(assert (fp.isNegative f))")


def test_quantifiers_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="declared"):
        _solve(
            "(declare-const x (_ BitVec 8))\n"
            "(assert (forall ((y (_ BitVec 8))) (= x y)))"
        )


def test_oversized_width_is_a_resource_boundary() -> None:
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        _solve("(declare-const x (_ BitVec 2048))\n(assert (= x x))")
    assert "width" in exc_info.value.errors()[0]["type"]


def test_other_fragments_still_reject_bitvectors() -> None:
    for logic in (SmtLogic.QF_LIA, SmtLogic.QF_LRA, SmtLogic.QF_UF):
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


def test_bitvector_logic_rejects_integer_terms() -> None:
    with pytest.raises(OperationDomainValidationError, match="declared"):
        _solve("(declare-const x Int)\n(assert (= x 0))")
