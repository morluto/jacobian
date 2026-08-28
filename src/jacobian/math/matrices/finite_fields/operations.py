"""Native exact operations over an explicit prime field."""

from jacobian.math.matrices.finite_fields import linear_algebra
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

__all__ = ["matrix_nullspace", "matrix_rank", "matrix_rref"]


def matrix_rank(matrix: PrimeFieldMatrix) -> int:
    """Return the rank of one canonical prime-field matrix."""

    return linear_algebra.rank(matrix)


def matrix_rref(
    matrix: PrimeFieldMatrix,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return the canonical reduced row-echelon form and pivot columns."""

    return linear_algebra.rref(matrix)


def matrix_nullspace(matrix: PrimeFieldMatrix) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic basis of the canonical right nullspace."""

    return linear_algebra.nullspace(matrix)
