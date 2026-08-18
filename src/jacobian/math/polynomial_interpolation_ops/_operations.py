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


def _parse(s: str) -> sympy.Expr:
    return sympy.sympify(s)


def compute_divided_differences(
    request: DividedDifferencesRequest,
) -> DividedDifferencesResult:
    """Compute Newton divided differences from sample points."""
    n = len(request.nodes)
    nodes = [_parse(x) for x in request.nodes]
    values = [_parse(v) for v in request.values]
    table = [list(values)]
    for j in range(1, n):
        row = []
        for i in range(n - j):
            numerator = table[j - 1][i + 1] - table[j - 1][i]
            denominator = nodes[i + j] - nodes[i]
            row.append(numerator / denominator)
        table.append(row)
    coeffs = tuple(str(sympy.simplify(table[j][0])) for j in range(n))
    return DividedDifferencesResult(coefficients=coeffs)


def compute_newton_form(request: NewtonFormRequest) -> NewtonFormResult:
    """Compute Newton form coefficients (same as divided differences)."""
    n = len(request.nodes)
    nodes = [_parse(x) for x in request.nodes]
    values = [sympy.Rational(v) for v in request.values]
    table = [list(values)]
    for j in range(1, n):
        row = []
        for i in range(n - j):
            numerator = table[j - 1][i + 1] - table[j - 1][i]
            denominator = nodes[i + j] - nodes[i]
            row.append(numerator / denominator)
        table.append(row)
    coeffs = tuple(str(sympy.simplify(table[j][0])) for j in range(n))
    return NewtonFormResult(coefficients=coeffs, nodes=request.nodes)


def compute_newton_evaluate(request: NewtonEvaluateRequest) -> NewtonEvaluateResult:
    """Evaluate a polynomial in Newton form using Horner-like nesting."""
    n = len(request.nodes)
    nodes = [sympy.Rational(x) for x in request.nodes]
    values = [sympy.Rational(v) for v in request.values]
    table = [list(values)]
    for j in range(1, n):
        row = []
        for i in range(n - j):
            numerator = table[j - 1][i + 1] - table[j - 1][i]
            denominator = nodes[i + j] - nodes[i]
            row.append(numerator / denominator)
        table.append(row)
    coeffs = [table[j][0] for j in range(n)]
    x = sympy.sympify(request.evaluation_point)
    result = coeffs[n - 1]
    for j in range(n - 2, -1, -1):
        result = coeffs[j] + (x - nodes[j]) * result
    return NewtonEvaluateResult(result=str(sympy.simplify(result)))
