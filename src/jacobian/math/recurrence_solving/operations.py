"""Recurrence solving backed by SymPy."""

from __future__ import annotations

from fractions import Fraction

__all__ = ["closed_form", "find_recurrence"]


def find_recurrence(sequence):  # type: ignore[no-untyped-def]
    n = len(sequence)
    vals = [Fraction(s) for s in sequence]
    for order in range(1, n // 2 + 1):
        if n < 2 * order:
            continue
        found = True
        for i in range(order, n):
            pred = sum(vals[j] * vals[i - order + j] for j in range(order))
            if pred != vals[i]:
                found = False
                break
        if found and order > 0:
            coeffs = [str(v) for v in vals[:order]]
            return {"coefficients": tuple(coeffs), "order": order}
    return {"coefficients": (), "order": 0}


def closed_form(char_coeffs, initial_values):  # type: ignore[no-untyped-def]
    import sympy

    x = sympy.Symbol("x")
    n = sympy.Symbol("n")
    char_poly_coeffs = [sympy.Rational(c) for c in char_coeffs]
    char_poly = sum(
        c * x ** (len(char_poly_coeffs) - 1 - i) for i, c in enumerate(char_poly_coeffs)
    )
    roots = sympy.solve(char_poly, x)
    if len({str(r) for r in roots}) != len(roots):
        return {"expression": "closed form with repeated roots not supported"}
    init = [sympy.Rational(v) for v in initial_values]
    a = sympy.Matrix([[r**i for r in roots] for i in range(len(roots))])
    b = sympy.Matrix(init[: len(roots)])
    consts = a.solve(b)
    expr = sum(c * r**n for c, r in zip(consts, roots, strict=True))
    return {"expression": str(sympy.simplify(expr))}
