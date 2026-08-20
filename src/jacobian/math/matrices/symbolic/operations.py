"""Exact symbolic matrix operations backed by SymPy.

The entries are canonical rational-function values over a declared variable
axis.  SymPy objects are constructed programmatically from validated sparse
terms; this module never parses caller text.
"""

from __future__ import annotations

from typing import Any

from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.values import RationalFunction

__all__ = ["symbolic_determinant", "symbolic_rank"]


def _matrix_from_values(
    entries: tuple[tuple[RationalFunction, ...], ...],
) -> Any:
    import sympy

    if not entries or not entries[0]:
        raise ValueError("symbolic matrix must be nonempty")
    rows = len(entries)
    columns = len(entries[0])
    if any(len(row) != columns for row in entries):
        raise ValueError("symbolic matrix rows must all have the same length")
    if rows > 8 or columns > 8:
        raise ValueError("symbolic matrix dimensions must be between 1 and 8")
    return sympy.Matrix(
        [[rational_function_to_sympy(entry) for entry in row] for row in entries]
    )


def symbolic_determinant(
    entries: tuple[tuple[RationalFunction, ...], ...],
    variables: tuple[str, ...],
) -> RationalFunction:
    """Return the determinant in the declared rational-function field."""
    matrix = _matrix_from_values(entries)
    if matrix.rows != matrix.cols:
        raise ValueError("determinant requires a square matrix")
    return rational_function_from_sympy(matrix.det(method="bareiss"), variables)


def symbolic_rank(
    entries: tuple[tuple[RationalFunction, ...], ...],
    variables: tuple[str, ...],
) -> tuple[int, tuple[int, ...]]:
    """Return the exact symbolic rank and RREF pivot columns."""
    del variables
    matrix = _matrix_from_values(entries)
    _, pivots = matrix.rref()
    return len(pivots), tuple(int(c) for c in pivots)


def symbolic_characteristic_polynomial(
    entries: tuple[tuple[RationalFunction, ...], ...],
    variables: tuple[str, ...],
) -> tuple[int, tuple[RationalFunction, ...]]:
    """Return (degree, descending coefficients) of det(lambda I - A)."""
    import sympy

    matrix = _matrix_from_values(entries)
    if matrix.rows != matrix.cols:
        raise ValueError("characteristic polynomial requires a square matrix")
    lam = sympy.Symbol("lambda")
    poly = (sympy.eye(matrix.rows) * lam - matrix).det(method="bareiss")
    expanded = sympy.Poly(poly, lam)
    coeffs = expanded.all_coeffs()
    return int(expanded.degree()), tuple(
        rational_function_from_sympy(coefficient, variables) for coefficient in coeffs
    )


def symbolic_eigenvalues(
    entries: tuple[tuple[RationalFunction, ...], ...],
    variables: tuple[str, ...],
) -> list[tuple[str, int]]:
    """Return a list of (eigenvalue_string, multiplicity) pairs."""
    from sympy import sstr

    del variables
    matrix = _matrix_from_values(entries)
    if matrix.rows != matrix.cols:
        raise ValueError("eigenvalues require a square matrix")
    eigenvalues = matrix.eigenvals()
    return [(sstr(value), int(mult)) for value, mult in eigenvalues.items()]
