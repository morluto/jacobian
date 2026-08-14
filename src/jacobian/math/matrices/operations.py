"""Exact matrix operations on canonical SymPy matrix inputs.

This is the supported public API for ``jacobian.math.matrices``.  Domain
adapters convert contract types to SymPy matrices, call these functions, and
convert results back.  The SymPy backend is private to this module and loaded
lazily so importing ``jacobian.math`` does not eagerly load packaged backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from sympy.matrices.matrixbase import MatrixBase

__all__ = [
    "adjugate",
    "characteristic_polynomial",
    "determinant",
    "inverse",
    "multiply",
    "rank",
    "rref",
    "smith_normal_form",
    "solve_linear_system",
    "trace",
]


def _exact_matrix(value: MatrixBase, *, maximum_dimension: int = 32) -> MatrixBase:
    import sympy
    from sympy.matrices.matrixbase import MatrixBase

    if not isinstance(value, MatrixBase):
        raise TypeError("matrix must be a SymPy MatrixBase")
    if (
        not 1 <= value.rows <= maximum_dimension
        or not 1 <= value.cols <= maximum_dimension
    ):
        raise ValueError(f"matrix dimensions must be between 1 and {maximum_dimension}")
    if any(not entry.is_number or entry.is_finite is not True for entry in value):
        raise ValueError("matrix entries must be finite exact numbers")
    if any(entry.has(sympy.Float) for entry in value):
        raise ValueError("matrix entries must be exact; SymPy Float is not supported")
    return value


def rref(matrix: MatrixBase) -> tuple[MatrixBase, tuple[int, ...]]:
    reduced, pivots = _exact_matrix(matrix).rref()
    return reduced, tuple(int(pivot) for pivot in pivots)


def inverse(matrix: MatrixBase) -> MatrixBase:
    source = _exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("inverse requires a square matrix")
    if source.det() == 0:
        raise ValueError("matrix is singular; inverse does not exist")
    return source.inv()


def trace(matrix: MatrixBase) -> Any:
    import sympy

    source = _exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("trace requires a square matrix")
    return sympy.simplify(source.trace())


def characteristic_polynomial(matrix: MatrixBase, variable: str) -> Any:
    source = _exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("characteristic polynomial requires a square matrix")
    return source.charpoly(variable)


def determinant(matrix: MatrixBase) -> Any:
    source = _exact_matrix(matrix, maximum_dimension=64)
    if source.rows != source.cols:
        raise ValueError("determinant requires a square matrix")
    return source.det(method="bareiss")


def rank(matrix: MatrixBase) -> tuple[int, tuple[int, ...]]:
    _, pivots = rref(matrix)
    return len(pivots), pivots


def smith_normal_form(matrix: MatrixBase) -> MatrixBase:
    import sympy
    from sympy.matrices.normalforms import smith_normal_form as sympy_smith_normal_form

    return sympy_smith_normal_form(_exact_matrix(matrix), domain=sympy.ZZ)


def multiply(left: MatrixBase, right: MatrixBase) -> MatrixBase:
    return _exact_matrix(left) * _exact_matrix(right)


def solve_linear_system(
    matrix: MatrixBase,
    right_hand_side: MatrixBase,
) -> tuple[MatrixBase, MatrixBase]:
    return cast(
        tuple[Any, Any],
        _exact_matrix(matrix).gauss_jordan_solve(_exact_matrix(right_hand_side)),
    )


def adjugate(matrix: MatrixBase) -> MatrixBase:
    source = _exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("adjugate requires a square matrix")
    return source.adjugate()
