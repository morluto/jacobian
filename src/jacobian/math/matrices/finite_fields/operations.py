"""Native exact operations over an explicit prime field."""

from jacobian.math.matrices.finite_fields import linear_algebra
from jacobian.math.matrices.finite_fields._models import (
    PrimeFieldMatrixRankResult,
    PrimeFieldRrefResult,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

__all__ = [
    "matrix_nullspace",
    "matrix_rank",
    "matrix_rref",
    "verify_rank",
    "verify_rref",
]


def verify_rank(claim: PrimeFieldMatrixRankResult) -> bool:
    """Check a serialized rank claim against its retained matrix.

    The source bounds the elimination by 1024 cubed field operations;
    primality is admitted by the same native path as rank production.
    """
    return matrix_rank(claim.source.matrix) == claim.rank


def verify_rref(claim: PrimeFieldRrefResult) -> bool:
    """Check the unique reduced form, pivots, and rank against the source.

    One admitted elimination uses the source's bounded matrix envelope.
    Neither checker is invoked by result construction or deserialization.
    """
    rows, pivots = matrix_rref(claim.source.matrix)
    return (
        rows == claim.rref_matrix.entries
        and pivots == claim.pivot_columns
        and len(pivots) == claim.rank
    )


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
