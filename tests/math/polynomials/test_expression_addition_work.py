"""Addition must charge the accumulated-clone work performed by ``_add``.

``_add`` copies the whole accumulated coefficient dictionary before merging each
operand, so an ``ADD`` whose first operand has large support and whose remaining
operands are scalars performs one full support copy per operand. The previous
estimate charged each operand's support once and omitted those repeated copies.
"""

from __future__ import annotations

from typing import Any

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._expression_normalize import (
    PolynomialAdd,
    PolynomialExpressionNormalizeRequest,
    PolynomialExpressionSource,
    PolynomialLiteral,
    PolynomialPower,
    PolynomialVariableExpression,
    _bounded_sum,
    _metrics,
    _support_bound,
    normalize_polynomial_expression,
)
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_TERMS


def _variables(count: int) -> tuple[str, ...]:
    return tuple(f"x{index}" for index in range(count))


def _power_of_sum(variables: tuple[str, ...], exponent: int):
    return PolynomialPower(
        base=PolynomialAdd(
            operands=tuple(
                PolynomialVariableExpression(name=name) for name in variables
            )
        ),
        exponent=exponent,
    )


def _literal_one() -> PolynomialLiteral:
    return PolynomialLiteral(value=CanonicalRational(num=1, den=1))


def test_addition_charges_accumulated_clone_work() -> None:
    """The reported work grows with the running support of the union.

    ``(x0+x1+x2+x3)**12`` then nested additions of 16 scalars: each level
    copies an accumulated dictionary whose support is far larger than one
    scalar. The charge must equal that clone-plus-merge total, not the
    single-pass sum of operand supports.
    """
    variables = _variables(4)
    variable_set = frozenset(variables)
    expression = _power_of_sum(variables, 12)
    for _ in range(4):
        expression = PolynomialAdd(
            operands=(expression, *(_literal_one() for _ in range(16)))
        )
    child_metrics = [_metrics(child) for child in expression.operands]
    accumulated_terms = 0
    accumulated_degree = 0
    accumulated_support = 0
    clone_work = 0
    for child in child_metrics:
        clone_work += accumulated_support
        accumulated_terms = _bounded_sum(
            (accumulated_terms, child.support), MAX_POLYNOMIAL_TERMS
        )
        accumulated_degree = max(accumulated_degree, child.degree)
        accumulated_support = _support_bound(
            accumulated_terms, accumulated_degree, variable_set
        )
    base_work = sum(child.work + child.expansion_terms for child in child_metrics)
    expected = base_work + clone_work
    single_pass = sum(child.support for child in child_metrics) + base_work
    assert single_pass < expected
    assert _metrics(expression).work == expected


def _request(expression: dict[str, Any], variables: tuple[str, ...]):
    return PolynomialExpressionNormalizeRequest.model_validate(
        {
            "coefficient_domain": "ZZ",
            "variables": list(variables),
            "expression": expression,
        }
    )


def _nested_add_scalars(
    base_expression: dict[str, Any], *, levels: int, scalars: int
) -> dict[str, Any]:
    literal = {"kind": "LITERAL", "value": {"num": 1, "den": 1}}
    expression = base_expression
    for _ in range(levels):
        expression = {
            "kind": "ADD",
            "operands": [expression, *[literal for _ in range(scalars)]],
        }
    return expression


def test_nested_scalar_additions_stay_exactly_decidable() -> None:
    """The corrected charge still admits a cheap accumulated-clone request."""
    variables = _variables(4)
    base = {
        "kind": "POWER",
        "base": {
            "kind": "ADD",
            "operands": [{"kind": "VARIABLE", "name": name} for name in variables],
        },
        "exponent": 12,
    }
    expression = _nested_add_scalars(base, levels=4, scalars=16)
    request = _request(expression, variables)
    result = normalize_polynomial_expression(
        PolynomialExpressionSource(
            coefficient_domain=request.coefficient_domain,
            variables=request.variables,
            expression=request.expression,
        )
    )
    assert result.polynomial.polynomial.terms


def test_addition_work_bound_rejects_an_over_budget_clone_chain() -> None:
    """An ADD tree whose clone charge exceeds the bound is refused."""
    variables = _variables(4)
    base = {
        "kind": "POWER",
        "base": {
            "kind": "ADD",
            "operands": [{"kind": "VARIABLE", "name": name} for name in variables],
        },
        "exponent": 28,
    }
    expression = _nested_add_scalars(base, levels=10, scalars=16)
    request = _request(expression, variables)
    with pytest.raises(OperationResourceAdmissionError):
        normalize_polynomial_expression(
            PolynomialExpressionSource(
                coefficient_domain=request.coefficient_domain,
                variables=request.variables,
                expression=request.expression,
            )
        )
