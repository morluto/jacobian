"""Domain functions for symmetric function operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.combinatorics.symmetric_functions._models import (
    IntegerPartition,
    PartitionConjugateResult,
    PartitionRequest,
    SchurExpansionRequest,
    SchurExpansionResult,
)


def compute_partition_conjugate(request: PartitionRequest) -> PartitionConjugateResult:
    """Compute the conjugate (transpose) of an integer partition."""

    parts = request.partition.parts
    if not parts:
        return PartitionConjugateResult(conjugate=IntegerPartition(parts=()))
    max_part = parts[0]
    conjugate = tuple(sum(1 for p in parts if p >= i) for i in range(1, max_part + 1))
    return PartitionConjugateResult(conjugate=IntegerPartition(parts=conjugate))


def _complete_homogeneous(variables: list[int], k: int) -> int:
    """Compute the complete homogeneous symmetric polynomial h_k at a point.

    Uses the recurrence h_k(x_1,...,x_n) = h_k(x_1,...,x_{n-1}) + x_n * h_{k-1}(x_1,...,x_n),
    which requires forward iteration in the DP.
    """

    if k == 0:
        return 1
    if k < 0:
        return 0
    dp: list[int] = [0] * (k + 1)
    dp[0] = 1
    for v in variables:
        for j in range(1, k + 1):
            dp[j] = dp[j] + v * dp[j - 1]
    return dp[k]


def compute_schur_evaluation(request: SchurExpansionRequest) -> SchurExpansionResult:
    """Evaluate a Schur function s_lambda at a point using the Jacobi-Trudi formula.

    s_lambda = det(h_{lambda_i - i + j}) where h_k is the complete homogeneous
    symmetric polynomial of degree k, and indices i, j range over the partition length.
    """

    partition = list(request.partition.parts)
    n = len(partition)
    if not partition:
        return SchurExpansionResult(value=format_canonical_integer(1))

    point = list(request.point)

    def h(k: int) -> int:
        if k < 0:
            return 0
        return _complete_homogeneous(point, k)

    size = n
    matrix: list[list[int]] = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = h(partition[i] - (i + 1) + (j + 1))

    result = _determinant(matrix)
    return SchurExpansionResult(value=format_canonical_integer(result))


def _determinant(matrix: list[list[int]]) -> int:
    """Compute the determinant of a square integer matrix via SymPy."""

    n = len(matrix)
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    from sympy import Matrix

    return int(Matrix(matrix).det())


__all__ = [
    "compute_partition_conjugate",
    "compute_schur_evaluation",
]
