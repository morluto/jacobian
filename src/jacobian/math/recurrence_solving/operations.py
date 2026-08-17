"""Recurrence solving backed by SymPy."""

from __future__ import annotations

from jacobian.canonical import parse_canonical_integer

__all__ = ["closed_form", "find_recurrence"]


def find_recurrence(sequence):  # type: ignore[no-untyped-def]
    import sympy

    values = [sympy.Rational(parse_canonical_integer(value)) for value in sequence]
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
            "status": "FOUND",
        }
    return {
        "coefficients": (),
        "order": 0,
        "status": "NO_FITTING_RECURRENCE",
    }


def closed_form(char_coeffs, initial_values):  # type: ignore[no-untyped-def]
    import sympy

    x = sympy.Symbol("x")
    n = sympy.Symbol("n", integer=True, nonnegative=True)
    char_poly_coeffs = [sympy.Rational(parse_canonical_integer(c)) for c in char_coeffs]
    char_poly = sum(
        c * x ** (len(char_poly_coeffs) - 1 - i) for i, c in enumerate(char_poly_coeffs)
    )
    roots = sympy.Poly(char_poly, x).all_roots()
    zero_root_multiplicity = sum(root == 0 for root in roots)
    nonzero_roots = list(dict.fromkeys(root for root in roots if root != 0))
    basis = [sympy.KroneckerDelta(index, n) for index in range(zero_root_multiplicity)]
    basis.extend(
        n**power * root**n
        for root in nonzero_roots
        for power in range(roots.count(root))
    )
    init = [sympy.Rational(parse_canonical_integer(v)) for v in initial_values]
    a = sympy.Matrix([[term.subs(n, i) for term in basis] for i in range(len(basis))])
    b = sympy.Matrix(init)
    consts = a.solve(b)
    expr = sum(c * term for c, term in zip(consts, basis, strict=True))
    return {"expression": str(sympy.simplify(expr))}
