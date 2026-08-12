"""Exact matrix operations on canonical SymPy matrix inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def adjugate(matrix: MatrixBase) -> MatrixBase:
    """Return the classical adjugate of a square matrix."""

    from jacobian.math.matrices import _sympy

    return _sympy.adjugate(matrix)


def characteristic_polynomial(matrix: MatrixBase, variable: str) -> Any:
    """Return the characteristic polynomial in the named variable."""

    from jacobian.math.matrices import _sympy

    return _sympy.characteristic_polynomial(matrix, variable)


def determinant(matrix: MatrixBase) -> Any:
    """Return the exact determinant of a square matrix."""

    from jacobian.math.matrices import _sympy

    return _sympy.determinant(matrix)


def rank(matrix: MatrixBase) -> tuple[int, tuple[int, ...]]:
    """Return exact rank together with the RREF pivot columns."""

    from jacobian.math.matrices import _sympy

    return _sympy.rank(matrix)


def rref(matrix: MatrixBase) -> tuple[MatrixBase, tuple[int, ...]]:
    """Return exact reduced row-echelon form and pivot columns."""

    from jacobian.math.matrices import _sympy

    return _sympy.rref(matrix)


def inverse(matrix: MatrixBase) -> MatrixBase:
    """Return the exact inverse of a square, nonsingular matrix."""

    from jacobian.math.matrices import _sympy

    return _sympy.inverse(matrix)


def multiply(left: MatrixBase, right: MatrixBase) -> MatrixBase:
    """Return the exact row-by-column matrix product."""

    from jacobian.math.matrices import _sympy

    return _sympy.matrix_product(left, right)


def smith_normal_form(matrix: MatrixBase) -> MatrixBase:
    """Return the Smith normal form of an exact integer matrix."""

    from jacobian.math.matrices import _sympy

    return _sympy.smith_normal_form(matrix)


def solve_linear_system(
    matrix: MatrixBase,
    right_hand_side: MatrixBase,
) -> tuple[MatrixBase, MatrixBase]:
    """Solve an exact linear system, returning solution and parameters."""

    from jacobian.math.matrices import _sympy

    return _sympy.rational_linear_solve(matrix, right_hand_side)


def trace(matrix: MatrixBase) -> Any:
    """Return the exact trace of a square matrix."""

    from jacobian.math.matrices import _sympy

    return _sympy.trace(matrix)
