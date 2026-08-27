"""Exact fixed-length simple path profile kernel."""

from __future__ import annotations

from jacobian.math.graphs.transforms._path_profile_models import (
    PathProfileRequest,
    PathProfileResult,
    PathProfileRow,
)


def compute_path_profile(request: PathProfileRequest) -> PathProfileResult:
    """For each ordered pair (source, target), count simple paths of the given length.

    Uses depth-first search with backtracking. For path_length=0, each vertex
    has a trivial path to itself. For path_length=1, adjacency determines counts.
    """
    graph = request.graph
    length = request.path_length
    vertices = list(graph.vertices)

    adj: dict[str, set[str]] = {v: set() for v in vertices}
    for edge in graph.edges:
        u, v = edge[0], edge[1]
        if u in adj:
            adj[u].add(v)
        if v in adj:
            adj[v].add(u)

    rows: list[PathProfileRow] = []
    for source in vertices:
        for target in vertices:
            count = _count_paths(source, target, length, adj, set())
            if count > 0:
                rows.append(
                    PathProfileRow(source=source, target=target, path_count=count)
                )

    return PathProfileResult(
        source=graph,
        path_length=length,
        rows=rows,
    )


def _count_paths(
    current: str,
    target: str,
    remaining: int,
    adj: dict[str, set[str]],
    visited: set[str],
) -> int:
    """Count simple paths of length `remaining` from current to target."""
    if remaining == 0:
        return 1 if current == target else 0
    visited = visited | {current}
    total = 0
    for neighbor in adj.get(current, set()):
        if neighbor not in visited:
            total += _count_paths(neighbor, target, remaining - 1, adj, visited)
    return total


__all__ = ["compute_path_profile"]
