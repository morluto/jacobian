"""Domain functions for quiver and path algebra operations."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.quivers._models import (
    AdjacencyMatricesResult,
    FiniteQuiver,
    FixedLengthPathsResult,
    VertexProfilesResult,
)
from jacobian.math.graphs.quivers._path_bounds import fixed_length_paths_envelope
from jacobian.math.matrices.values import IntegerMatrix


def _integer_matrix(entries: list[list[int]]) -> IntegerMatrix:
    return IntegerMatrix(
        row_count=len(entries),
        column_count=len(entries[0]) if entries else 0,
        entries=tuple(tuple(value for value in row) for row in entries)
    )


def adjacency_matrices(quiver: FiniteQuiver) -> AdjacencyMatricesResult:
    """Compute the adjacency matrix and its transpose."""
    n = quiver.vertex_count
    matrix = [[0] * n for _ in range(n)]
    for source, target in quiver.arrows:
        matrix[source][target] += 1
    transpose = [[matrix[j][i] for j in range(n)] for i in range(n)]
    return AdjacencyMatricesResult(
        quiver=quiver,
        adjacency_matrix=_integer_matrix(matrix),
        transpose_matrix=_integer_matrix(transpose),
    )


def vertex_profiles(quiver: FiniteQuiver) -> VertexProfilesResult:
    """Compute in-degree and out-degree for each vertex."""
    n = quiver.vertex_count
    in_degrees = [0] * n
    out_degrees = [0] * n
    for source, target in quiver.arrows:
        out_degrees[source] += 1
        in_degrees[target] += 1
    return VertexProfilesResult(
        quiver=quiver,
        in_degrees=tuple(in_degrees),
        out_degrees=tuple(out_degrees),
    )


def fixed_length_paths(quiver: FiniteQuiver, length: int) -> FixedLengthPathsResult:
    """Count paths of fixed length between all vertex pairs using matrix powers."""
    try:
        fixed_length_paths_envelope(
            vertex_count=quiver.vertex_count,
            arrow_count=len(quiver.arrows),
            length=length,
        )
    except ValueError as error:
        raise OperationDomainValidationError(
            location=("quiver", "length"),
            code="quiver.fixed_length_paths_exceeds_envelope",
            message=str(error),
        ) from error
    n = quiver.vertex_count
    matrix = [[0] * n for _ in range(n)]
    for source, target in quiver.arrows:
        matrix[source][target] += 1

    if length == 0:
        result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    else:
        result = matrix
        for _ in range(length - 1):
            result = _matrix_multiply(result, matrix)

    total = sum(sum(row) for row in result)
    return FixedLengthPathsResult(
        quiver=quiver,
        length=length,
        path_matrix=_integer_matrix(result),
        total_paths=total,
    )


def _matrix_multiply(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    n = len(a)
    m = len(b[0])
    k = len(b)
    result = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for _l in range(k):
                result[i][j] += a[i][_l] * b[_l][j]
    return result


def verify_adjacency_matrices(claim: AdjacencyMatricesResult) -> bool:
    """Verify both retained adjacency relations against the source quiver."""
    try:
        expected = adjacency_matrices(claim.quiver)
        return (
            claim.adjacency_matrix == expected.adjacency_matrix
            and claim.transpose_matrix == expected.transpose_matrix
        )
    except (TypeError, ValueError):
        return False


def verify_fixed_length_paths(
    claim: FixedLengthPathsResult, length: int | None = None
) -> bool:
    """Verify a path-count matrix against its source quiver and length."""
    try:
        expected = fixed_length_paths(
            claim.quiver, claim.length if length is None else length
        )
        return (
            claim.path_matrix == expected.path_matrix
            and claim.total_paths == expected.total_paths
        )
    except (TypeError, ValueError):
        return False


__all__ = [
    "adjacency_matrices",
    "fixed_length_paths",
    "verify_adjacency_matrices",
    "verify_fixed_length_paths",
    "vertex_profiles",
]
