"""Exact linear algebra on SymPy matrices."""

from sympy.matrices.matrixbase import MatrixBase

from jacobian.domains.matrix_lattice import kernels

__all__ = ["inverse", "rref", "trace"]


def rref(matrix: MatrixBase) -> tuple[MatrixBase, tuple[int, ...]]:
    """Return exact reduced row-echelon form and pivot columns."""

    return kernels.rref(matrix)


def inverse(matrix: MatrixBase) -> MatrixBase:
    """Return the exact inverse of a square, nonsingular matrix."""

    return kernels.inverse(matrix)


def trace(matrix: MatrixBase) -> object:
    """Return the exact trace of a square matrix."""

    return kernels.trace(matrix)
