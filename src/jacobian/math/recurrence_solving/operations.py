"""Recurrence solving backed by SymPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jacobian._exact import CanonicalRational

__all__ = ["ClosedForm", "Recurrence", "closed_form", "find_recurrence"]


@dataclass(frozen=True, slots=True)
class Recurrence:
    coefficients: tuple[CanonicalRational, ...]
    order: int
    status: Literal["FOUND", "NO_FITTING_RECURRENCE"]


@dataclass(frozen=True, slots=True)
class ClosedForm:
    expression: str


def find_recurrence(sequence: tuple[CanonicalRational, ...]) -> Recurrence:
    import sympy

    from jacobian.math.recurrence_solving._models import RecurrenceFindRequest

    request = RecurrenceFindRequest(sequence=sequence)

    values = [sympy.Rational(*value.as_integer_ratio()) for value in request.sequence]
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
        return Recurrence(
            coefficients=tuple(
                CanonicalRational.from_integer_ratio(int(value.p), int(value.q))
                for value in solution
            ),
            order=order,
            status="FOUND",
        )
    return Recurrence(coefficients=(), order=0, status="NO_FITTING_RECURRENCE")


def closed_form(
    char_coeffs: tuple[CanonicalRational, ...],
    initial_values: tuple[CanonicalRational, ...],
) -> ClosedForm:
    import sympy

    from jacobian.math.recurrence_solving._models import ClosedFormRequest

    request = ClosedFormRequest(
        characteristic_coefficients=char_coeffs,
        initial_values=initial_values,
    )

    x = sympy.Symbol("x")
    n = sympy.Symbol("n", integer=True, nonnegative=True)
    char_poly_coeffs = [
        sympy.Rational(*value.as_integer_ratio())
        for value in request.characteristic_coefficients
    ]
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
    init = [
        sympy.Rational(*value.as_integer_ratio()) for value in request.initial_values
    ]
    a = sympy.Matrix([[term.subs(n, i) for term in basis] for i in range(len(basis))])
    b = sympy.Matrix(init)
    consts = a.solve(b)
    expr = sum(c * term for c, term in zip(consts, basis, strict=True))
    return ClosedForm(expression=str(sympy.simplify(expr)))
