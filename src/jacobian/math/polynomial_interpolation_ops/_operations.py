"""Domain functions for polynomial interpolation operations."""

from __future__ import annotations

import sympy

from jacobian.math.polynomial_interpolation_ops._models import (
    DividedDifferencesRequest,
    DividedDifferencesResult,
    NewtonEvaluateRequest,
    NewtonEvaluateResult,
    NewtonFormRequest,
    NewtonFormResult,
)


def _parse_rational(s: str) -> sympy.Rational:
    """Parse a canonical rational string into an exact sympy.Rational."""
    return sympy.Rational(s)


def _divided_differences(
    nodes: list[sympy.Rational], values: list[sympy.Rational]
) -> list[list[sympy.Rational]]:
    """Compute the full divided-difference table."""
    n = len(nodes)
    table = [list(values)]
    for j in range(1, n):
        row = []
        for i in range(n - j):
            numerator = table[j - 1][i + 1] - table[j - 1][i]
            denominator = nodes[i + j] - nodes[i]
            row.append(numerator / denominator)
        table.append(row)
    return table


def compute_divided_differences(
    request: DividedDifferencesRequest,
) -> DividedDifferencesResult:
    """Compute Newton divided differences from sample points."""
    nodes = [_parse_rational(x) for x in request.nodes]
    values = [_parse_rational(v) for v in request.values]
    table = _divided_differences(nodes, values)
    n = len(nodes)
    coeffs = tuple(str(sympy.simplify(table[j][0])) for j in range(n))
    return DividedDifferencesResult(coefficients=coeffs)


def compute_newton_form(request: NewtonFormRequest) -> NewtonFormResult:
    """Compute Newton form coefficients (same as divided differences)."""
    nodes = [_parse_rational(x) for x in request.nodes]
    values = [_parse_rational(v) for v in request.values]
    table = _divided_differences(nodes, values)
    n = len(nodes)
    coeffs = tuple(str(sympy.simplify(table[j][0])) for j in range(n))
    return NewtonFormResult(coefficients=coeffs, nodes=request.nodes)


def compute_newton_evaluate(request: NewtonEvaluateRequest) -> NewtonEvaluateResult:
    """Evaluate a polynomial in Newton form using Horner-like nesting."""
    nodes = [_parse_rational(x) for x in request.nodes]
    values = [_parse_rational(v) for v in request.values]
    table = _divided_differences(nodes, values)
    n = len(nodes)
    coeffs = [table[j][0] for j in range(n)]
    x = _parse_rational(request.evaluation_point)
    result = coeffs[n - 1]
    for j in range(n - 2, -1, -1):
        result = coeffs[j] + (x - nodes[j]) * result
    return NewtonEvaluateResult(result=str(sympy.simplify(result)))
