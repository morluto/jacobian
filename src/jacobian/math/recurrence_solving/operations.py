"""Recurrence solving backed by SymPy."""

from __future__ import annotations

__all__ = ["closed_form", "find_recurrence"]


def find_recurrence(sequence):  # type: ignore[no-untyped-def]
    import sympy

    values = [sympy.Rational(value) for value in sequence]
    for order in range(1, len(values)):
        coefficient_matrix = sympy.Matrix(
            [
                [values[index - offset] for offset in range(1, order + 1)]
                for index in range(order, len(values))
            ]
        )
        targets = sympy.Matrix(values[order:])
        try:
            solution, parameters = coefficient_matrix.gauss_jordan_solve(targets)
        except ValueError:
            continue
        if parameters:
            solution = solution.subs(dict.fromkeys(parameters, 0))
        return {
            "coefficients": tuple(str(value) for value in solution),
            "order": order,
        }
    return {"coefficients": (), "order": 0}


def closed_form(char_coeffs, initial_values):  # type: ignore[no-untyped-def]
    import sympy

    x = sympy.Symbol("x")
    n = sympy.Symbol("n")
    char_poly_coeffs = [sympy.Rational(c) for c in char_coeffs]
    char_poly = sum(
        c * x ** (len(char_poly_coeffs) - 1 - i) for i, c in enumerate(char_poly_coeffs)
    )
    roots = sympy.roots(char_poly, x)
    if sum(roots.values()) != len(initial_values):
        raise ValueError("characteristic polynomial degree must match initial values")
    basis = [
        n**power * root**n
        for root, multiplicity in roots.items()
        for power in range(multiplicity)
    ]
    init = [sympy.Rational(v) for v in initial_values]
    a = sympy.Matrix([[term.subs(n, i) for term in basis] for i in range(len(basis))])
    b = sympy.Matrix(init)
    consts = a.solve(b)
    expr = sum(c * term for c, term in zip(consts, basis, strict=True))
    return {"expression": str(sympy.simplify(expr))}
