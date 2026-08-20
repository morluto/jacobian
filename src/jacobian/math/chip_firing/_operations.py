"""Exact chip-firing operations."""

from __future__ import annotations

from jacobian.math.chip_firing._models import (
    FiringRequest,
    FiringResult,
    LaplacianRequest,
    LaplacianResult,
)


def compute_laplacian(request: LaplacianRequest) -> LaplacianResult:
    """Compute the graph Laplacian L = D - A where D is the degree matrix."""
    vertices = request.graph.vertices
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}

    # Build adjacency matrix
    adj = [[0] * n for _ in range(n)]
    for u, v in request.graph.edges:
        i, j = idx[u], idx[v]
        adj[i][j] += 1
        adj[j][i] += 1

    # Laplacian: L[i][i] = degree(i), L[i][j] = -adj[i][j]
    laplacian = []
    degrees = []
    for i in range(n):
        deg = sum(adj[i])
        degrees.append(deg)
        row = []
        for j in range(n):
            if i == j:
                row.append(deg)
            else:
                row.append(-adj[i][j])
        laplacian.append(tuple(row))

    return LaplacianResult(
        vertices=vertices,
        laplacian=tuple(laplacian),
        degrees=tuple(degrees),
    )


def compute_firing(request: FiringRequest) -> FiringResult:
    """Fire a vertex: D' = D - L*e_v where L is the Laplacian."""
    vertices = request.graph.vertices
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}

    # Build adjacency
    adj = [[0] * n for _ in range(n)]
    for u, v in request.graph.edges:
        i, j = idx[u], idx[v]
        adj[i][j] += 1
        adj[j][i] += 1

    fire_idx = idx[request.firing_vertex]
    result = list(request.divisor)

    # Firing: lose degree(v) chips from v, each neighbor gains 1
    deg = sum(adj[fire_idx])
    result[fire_idx] -= deg
    for j in range(n):
        if adj[fire_idx][j] > 0:
            result[j] += adj[fire_idx][j]

    return FiringResult(
        vertex=request.firing_vertex,
        fired_divisor=tuple(result),
    )
