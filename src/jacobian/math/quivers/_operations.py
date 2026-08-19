"""Domain functions for quiver and path algebra operations."""

from __future__ import annotations

from jacobian.math.quivers._models import (
    AdjacencyMatricesRequest,
    AdjacencyMatricesResult,
    FixedLengthPathsRequest,
    FixedLengthPathsResult,
    VertexProfilesRequest,
    VertexProfilesResult,
)


def compute_adjacency_matrices(
    request: AdjacencyMatricesRequest,
) -> AdjacencyMatricesResult:
    """Compute the adjacency matrix and its transpose."""
    n = request.quiver.vertex_count
    matrix = [[0] * n for _ in range(n)]
    for source, target in request.quiver.arrows:
        matrix[source][target] += 1
    adj = tuple(tuple(row) for row in matrix)
    transpose = tuple(tuple(matrix[j][i] for j in range(n)) for i in range(n))
    return AdjacencyMatricesResult(
        adjacency_matrix=adj,
        transpose_matrix=transpose,
        vertex_count=n,
    )


def compute_vertex_profiles(
    request: VertexProfilesRequest,
) -> VertexProfilesResult:
    """Compute in-degree and out-degree for each vertex."""
    n = request.quiver.vertex_count
    in_degrees = [0] * n
    out_degrees = [0] * n
    for source, target in request.quiver.arrows:
        out_degrees[source] += 1
        in_degrees[target] += 1
    return VertexProfilesResult(
        in_degrees=tuple(in_degrees),
        out_degrees=tuple(out_degrees),
        vertex_count=n,
    )


def compute_fixed_length_paths(
    request: FixedLengthPathsRequest,
) -> FixedLengthPathsResult:
    """Count paths of fixed length between all vertex pairs using matrix powers."""
    n = request.quiver.vertex_count
    matrix = [[0] * n for _ in range(n)]
    for source, target in request.quiver.arrows:
        matrix[source][target] += 1

    if request.length == 0:
        result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    else:
        result = matrix
        for _ in range(request.length - 1):
            result = _matrix_multiply(result, matrix)

    total = sum(sum(row) for row in result)
    return FixedLengthPathsResult(
        path_matrix=tuple(tuple(row) for row in result),
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
