"""Small exact linear-algebra facts shared by mathematical domains."""

from __future__ import annotations

from collections.abc import Sequence


def symmetric_inertia(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, int, int]:
    """Return positive, negative, and zero inertia of a symmetric matrix."""
    from sympy import Matrix, oo, symbols

    sympy_matrix = Matrix(matrix)
    dimension = sympy_matrix.rows
    variable = symbols("lambda")
    characteristic = sympy_matrix.charpoly(variable).as_poly()
    zero = dimension - int(sympy_matrix.rank())
    negative = sum(
        multiplicity * (int(factor.count_roots(-oo, 0)) - int(factor.eval(0) == 0))
        for factor, multiplicity in characteristic.sqf_list()[1]
    )
    return dimension - negative - zero, negative, zero


__all__ = ["symmetric_inertia"]
