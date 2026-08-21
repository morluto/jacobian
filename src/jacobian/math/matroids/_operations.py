"""Domain-owned linear matroid operations."""

from __future__ import annotations

from jacobian.math.matroids._models import (
    LinearMatroid,
    MatroidClosureRequest,
    MatroidClosureResult,
    MatroidRankRequest,
    MatroidRankResult,
)


def _gaussian_rank(matrix: list[list[int]], prime: int) -> int:
    """Compute the rank of a matrix over GF(prime) using Gaussian elimination."""
    rows = len(matrix)
    if rows == 0:
        return 0
    cols = len(matrix[0])
    augmented = [row[:] for row in matrix]

    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if augmented[r][col] % prime != 0:
                pivot = r
                break
        if pivot is None:
            continue
        augmented[rank], augmented[pivot] = augmented[pivot], augmented[rank]
        inv_pivot = pow(augmented[rank][col] % prime, prime - 2, prime)
        for c in range(cols):
            augmented[rank][c] = (augmented[rank][c] * inv_pivot) % prime
        for r in range(rows):
            if r == rank:
                continue
            factor = augmented[r][col] % prime
            if factor != 0:
                for c in range(cols):
                    augmented[r][c] = (augmented[r][c] - factor * augmented[rank][c]) % prime
        rank += 1
        if rank >= rows:
            break
    return rank


def _column_matrix(matroid: LinearMatroid, column_indices: list[int] | None = None) -> list[list[int]]:
    """Build a column-major matrix from selected columns of the matroid."""
    cols = column_indices if column_indices is not None else list(range(len(matroid.columns)))
    return [
        [matroid.columns[j][i] for j in cols]
        for i in range(matroid.num_rows)
    ]


def compute_rank(request: MatroidRankRequest) -> MatroidRankResult:
    """Compute the rank of a linear matroid over a prime field."""
    matrix = _column_matrix(request.matroid)
    rank = _gaussian_rank(matrix, request.matroid.prime)
    return MatroidRankResult(
        rank=rank,
        ground_size=len(request.matroid.columns),
    )


def compute_closure(request: MatroidClosureRequest) -> MatroidClosureResult:
    """Compute the closure (smallest flat) of a subset in a linear matroid.

    The closure of S is S union all elements e such that e is in the span of S.
    """
    matroid = request.matroid
    subset = list(request.subset)

    subset_rank = _gaussian_rank(_column_matrix(matroid, subset), matroid.prime)

    closure = set(subset)
    for i in range(len(matroid.columns)):
        if i in closure:
            continue
        test_subset = sorted(closure | {i})
        test_rank = _gaussian_rank(
            _column_matrix(matroid, test_subset), matroid.prime
        )
        if test_rank == subset_rank:
            closure.add(i)

    closure_sorted = sorted(closure)
    return MatroidClosureResult(
        closure=tuple(closure_sorted),
        closure_size=len(closure_sorted),
        rank=subset_rank,
    )


__all__ = [
    "compute_closure",
    "compute_rank",
]
