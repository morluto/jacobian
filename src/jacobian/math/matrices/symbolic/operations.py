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

__all__ = ["symbolic_determinant", "symbolic_linear_system_solve", "symbolic_rank"]


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


def symbolic_linear_system_solve(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
    variables: tuple[str, ...],
) -> tuple[
    str,
    tuple[RationalFunction, ...] | None,
    tuple[RationalFunction, ...] | None,
    tuple[tuple[RationalFunction, ...], ...] | None,
]:
    """Solve ``A x = b`` over ``QQ(t_1, ..., t_n)``.

    Returns ``(classification, solution, particular_solution, nullspace_basis)``.
    """
    import sympy

    matrix = _matrix_from_values(entries)
    rhs_vec = sympy.Matrix(
        [[rational_function_to_sympy(v) for v in rhs]]
    ).T

    rank_coeff = matrix.rank()
    aug = matrix.row_join(rhs_vec)
    rank_aug = aug.rank()
    n_cols = matrix.cols

    if rank_aug > rank_coeff:
        return "INCONSISTENT", None, None, None

    if rank_coeff == n_cols:
        # Unique solution
        try:
            sol = matrix.LUsolve(rhs_vec)
            solution = tuple(
                rational_function_from_sympy(sol[i], variables)
                for i in range(n_cols)
            )
            return "UNIQUE", solution, None, None
        except Exception:
            pass

    # Non-unique consistent system
    try:
        null = matrix.nullspace()
        nullspace_basis: tuple[tuple[RationalFunction, ...], ...] = tuple(
            tuple(
                rational_function_from_sympy(vec[i], variables)
                for i in range(n_cols)
            )
            for vec in null
        )

        # Find a particular solution using the pseudo-inverse approach
        # or least squares.  For now, use sympy's linear solve.
        # Use the augmented matrix rref to find a particular solution
        rref_mat, pivots = aug.rref()
        # The particular solution: set free variables to 0
        # Extract from the RREF of the augmented matrix
        particular = []
        for j in range(n_cols):
            particular.append(
                rational_function_from_sympy(sympy.Integer(0), variables)
            )

        for i, pivot_col in enumerate(pivots):
            if pivot_col < n_cols:
                particular[pivot_col] = rational_function_from_sympy(
                    rref_mat[i, n_cols], variables
                )

        return (
            "NON_UNIQUE",
            None,
            tuple(particular),
            nullspace_basis if nullspace_basis else None,
        )
    except Exception:
        pass

    return "INCONSISTENT", None, None, None
