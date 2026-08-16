"""Exact symbolic matrix operations backed by SymPy.

The entries are SymPy expressions over a declared list of variables,
interpreted as elements of the multivariate rational-function field
``QQ(t_1, ..., t_n)``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["symbolic_determinant", "symbolic_rank"]


def _parse_matrix(entries: list[list[str]], variables: list[str]) -> Any:
    import sympy

    if not entries or not entries[0]:
        raise ValueError("symbolic matrix must be nonempty")
    symbols = [sympy.Symbol(name) for name in variables]
    rows = len(entries)
    columns = len(entries[0])
    if any(len(row) != columns for row in entries):
        raise ValueError("symbolic matrix rows must all have the same length")
    if rows > 32 or columns > 32:
        raise ValueError("symbolic matrix dimensions must be between 1 and 32")
    local = dict(zip(variables, symbols, strict=True))
    matrix_rows = []
    for row in entries:
        matrix_rows.append([sympy.sympify(entry, locals=local) for entry in row])
    matrix = sympy.Matrix(matrix_rows)
    if any(entry.has(sympy.Float) for entry in matrix):
        raise ValueError("symbolic matrix entries must be exact; no SymPy Float")
    return matrix, symbols


def symbolic_determinant(
    entries: list[list[str]],
    variables: list[str],
) -> str:
    """Return the exact symbolic determinant as a canonical SymPy string."""
    matrix, _ = _parse_matrix(entries, variables)
    if matrix.rows != matrix.cols:
        raise ValueError("determinant requires a square matrix")
    return str(matrix.det(method="bareiss"))


def symbolic_rank(
    entries: list[list[str]],
    variables: list[str],
) -> tuple[int, tuple[int, ...]]:
    """Return the exact symbolic rank and RREF pivot columns."""
    matrix, _ = _parse_matrix(entries, variables)
    _, pivots = matrix.rref()
    return len(pivots), tuple(int(c) for c in pivots)


def symbolic_characteristic_polynomial(
    entries: list[list[str]],
    variables: list[str],
) -> tuple[int, list[str]]:
    """Return (degree, descending coefficients) of det(lambda I - A)."""
    import sympy

    matrix, _ = _parse_matrix(entries, variables)
    if matrix.rows != matrix.cols:
        raise ValueError("characteristic polynomial requires a square matrix")
    lam = sympy.Symbol("lambda")
    poly = (sympy.eye(matrix.rows) * lam - matrix).det(method="bareiss")
    expanded = sympy.Poly(poly, lam)
    coeffs = expanded.all_coeffs()
    return int(expanded.degree()), [str(c) for c in coeffs]


def symbolic_eigenvalues(
    entries: list[list[str]],
    variables: list[str],
) -> list[tuple[str, int]]:
    """Return a list of (eigenvalue_string, multiplicity) pairs."""
    matrix, _ = _parse_matrix(entries, variables)
    if matrix.rows != matrix.cols:
        raise ValueError("eigenvalues require a square matrix")
    eigenvalues = matrix.eigenvals()
    return [(str(value), int(mult)) for value, mult in eigenvalues.items()]
