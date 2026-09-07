"""Arithmetic admission survives constants mixed with variables and conditionals."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic._smt import SmtLogic, SmtSolveRequest, solve_smt


@pytest.mark.parametrize("base", ["(+ x 1)", "(ite p x 1)", "(- x 1)"])
def test_mixed_arithmetic_cannot_drop_its_growth_bound(base: str) -> None:
    expression = base
    for _ in range(24):
        expression = f"(* {'9' * 1000} {expression})"
    request = SmtSolveRequest(
        logic=SmtLogic.QF_LIA,
        smtlib=(
            "(set-logic QF_LIA)(declare-const x Int)(declare-const p Bool)"
            f"(assert (= {expression} 1))(check-sat)"
        ),
    )
    with pytest.raises(OperationDomainValidationError, match=r"growth|work"):
        solve_smt(request)


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("(ite true 1 2)", 1),
        ("(ite p 1 2)", 1),
        ("(+ (ite p x 1) 1)", 2),
        ("(mod (- 1) 100000)", 99999),
        ("(div (- 7) 3)", -3),
    ],
)
def test_small_conditional_and_signed_arithmetic_remains_accepted(
    expression: str, expected: int
) -> None:
    value = f"(- {-expected})" if expected < 0 else str(expected)
    result = solve_smt(
        SmtSolveRequest(
            logic=SmtLogic.QF_LIA,
            smtlib=(
                "(set-logic QF_LIA)(declare-const x Int)(declare-const p Bool)"
                f"(assert (= {expression} {value}))(check-sat)"
            ),
        )
    )
    assert result.outcome == "SAT"
