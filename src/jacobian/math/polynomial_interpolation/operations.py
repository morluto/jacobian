"""Exact polynomial interpolation kernels."""

from __future__ import annotations

from fractions import Fraction

__all__ = ["multipoint_evaluate", "newton_interpolation"]


def newton_interpolation(
    points: list[tuple[Fraction, Fraction]],
) -> tuple[list[Fraction], list[Fraction]]:
    """Newton-form interpolation via divided differences.

    Returns (coefficients, divided_differences) where coefficients are
    the standard polynomial coefficients [a_0, ..., a_n] and
    divided_differences are the Newton coefficients [f[x_0], f[x_0,x_1], ...].
    """
    n = len(points)
    xs = [p[0] for p in points]

    # Compute divided differences
    # table[i][j] = f[x_i, ..., x_{i+j}] (divided difference of order j)
    table = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = points[i][1]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (xs[i + j] - xs[i])

    div_diffs = [table[0][j] for j in range(n)]

    # Convert Newton form to standard coefficients
    # p(x) = sum_{k=0}^{n-1} div_diffs[k] * prod_{i=0}^{k-1} (x - x_i)
    coeffs = [Fraction(0)] * n
    term = [Fraction(0)] * n
    term[0] = Fraction(1)
    for k in range(n):
        for i in range(n):
            coeffs[i] += div_diffs[k] * term[i]
        if k < n - 1:
            new_term = [Fraction(0)] * n
            new_term[0] = -xs[k] * term[0]
            for i in range(1, n):
                new_term[i] = term[i - 1] - xs[k] * term[i]
            term = new_term

    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()

    return (coeffs, div_diffs)


def multipoint_evaluate(
    coefficients: list[Fraction],
    points: list[Fraction],
) -> list[Fraction]:
    """Evaluate a polynomial at multiple points using Horner's method."""
    results = []
    for x in points:
        result = Fraction(0)
        for coeff in reversed(coefficients):
            result = result * x + coeff
        results.append(result)
    return results
