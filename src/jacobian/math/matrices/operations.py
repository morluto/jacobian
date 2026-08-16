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
    "kronecker_product",
    "multiply",
    "partial_trace",
    "permanent",
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


def kronecker_product(left: MatrixBase, right: MatrixBase) -> MatrixBase:
    import sympy

    return sympy.kronecker_product(_exact_matrix(left), _exact_matrix(right))


def partial_trace(
    matrix: MatrixBase,
    traced_dimension: int,
    kept_dimension: int,
) -> MatrixBase:
    """Trace out the first traced_dimension factor of a Kronecker product.

    The composite matrix is the Kronecker product A (x) B of a
    traced_dimension x traced_dimension matrix A (the traced subsystem)
    by a kept_dimension x kept_dimension matrix B (the kept subsystem),
    stored as a block matrix in row-major block order.  The returned matrix is
    the trace over the traced factor, i.e. trace(A) * B.
    """

    import sympy

    source = _exact_matrix(matrix)
    total = traced_dimension * kept_dimension
    if source.rows != source.cols:
        raise ValueError("partial trace requires a square composite matrix")
    if source.rows != total:
        raise ValueError(
            "partial trace dimensions are inconsistent with the composite matrix"
        )
    if traced_dimension <= 0 or kept_dimension <= 0:
        raise ValueError("partial trace subsystem dimensions must be positive")
    accumulator = sympy.zeros(kept_dimension)
    for block in range(traced_dimension):
        block_row = block * kept_dimension
        block_col = block * kept_dimension
        accumulator = sympy.Matrix(
            [
                [
                    accumulator[i, j] + source[block_row + i, block_col + j]
                    for j in range(kept_dimension)
                ]
                for i in range(kept_dimension)
            ]
        )
    return accumulator


def permanent(matrix: MatrixBase) -> Any:
    from sympy import Permanent

    source = _exact_matrix(matrix, maximum_dimension=64)
    if source.rows != source.cols:
        raise ValueError("permanent requires a square matrix")
    return Permanent(source).doit()
