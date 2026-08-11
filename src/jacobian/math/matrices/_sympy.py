"""Private SymPy backend for exact matrix operations."""

from __future__ import annotations

from typing import Any, cast

import sympy
from sympy.matrices.matrixbase import MatrixBase


def exact_matrix(value: MatrixBase) -> MatrixBase:
    if not isinstance(value, MatrixBase):
        raise TypeError("matrix must be a SymPy MatrixBase")
    if not 1 <= value.rows <= 32 or not 1 <= value.cols <= 32:
        raise ValueError("matrix dimensions must be between 1 and 32")
    if any(not entry.is_number or entry.is_finite is not True for entry in value):
        raise ValueError("matrix entries must be finite exact numbers")
    if any(entry.has(sympy.Float) for entry in value):
        raise ValueError("matrix entries must be exact; SymPy Float is not supported")
    return value


def rref(matrix: MatrixBase) -> tuple[MatrixBase, tuple[int, ...]]:
    reduced, pivots = exact_matrix(matrix).rref()
    return reduced, tuple(int(pivot) for pivot in pivots)


def nullspace_rref(matrix: MatrixBase) -> tuple[MatrixBase, tuple[int, ...]]:
    return rref(matrix)


def inverse(matrix: MatrixBase) -> MatrixBase:
    source = exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("inverse requires a square matrix")
    if source.det() == 0:
        raise ValueError("matrix is singular; inverse does not exist")
    return source.inv()


def trace(matrix: MatrixBase) -> Any:
    source = exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("trace requires a square matrix")
    return sympy.simplify(source.trace())


def characteristic_polynomial(matrix: MatrixBase, variable: str) -> Any:
    source = exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("characteristic polynomial requires a square matrix")
    return source.charpoly(variable)


def determinant(matrix: MatrixBase) -> Any:
    source = exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("determinant requires a square matrix")
    return source.det(method="bareiss")


def rank(matrix: MatrixBase) -> tuple[int, tuple[int, ...]]:
    _, pivots = rref(matrix)
    return len(pivots), pivots


def smith_normal_form(matrix: MatrixBase) -> MatrixBase:
    source = exact_matrix(matrix)
    from sympy.matrices.normalforms import smith_normal_form as sympy_smith_normal_form

    return sympy_smith_normal_form(source, domain=sympy.ZZ)


def matrix_product(left: MatrixBase, right: MatrixBase) -> MatrixBase:
    return exact_matrix(left) * exact_matrix(right)


def rational_linear_solve(
    matrix: MatrixBase,
    rhs: MatrixBase,
) -> tuple[MatrixBase, MatrixBase]:
    return cast(
        tuple[MatrixBase, MatrixBase],
        exact_matrix(matrix).gauss_jordan_solve(exact_matrix(rhs)),
    )


def adjugate(matrix: MatrixBase) -> MatrixBase:
    source = exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("adjugate requires a square matrix")
    return source.adjugate()
