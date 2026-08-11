"""Exact matrix operations on canonical SymPy matrix inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sympy.matrices.matrixbase import MatrixBase

__all__ = ["determinant", "inverse", "rank", "rref", "trace"]


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


def trace(matrix: MatrixBase) -> Any:
    """Return the exact trace of a square matrix."""

    from jacobian.math.matrices import _sympy

    return _sympy.trace(matrix)
