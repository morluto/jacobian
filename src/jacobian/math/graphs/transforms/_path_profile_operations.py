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
        counts = _count_paths_by_endpoint(source, length, adj)
        for target in vertices:
            count = counts.get(target, 0)
            if count:
                rows.append(
                    PathProfileRow(source=source, target=target, path_count=count)
                )

    return PathProfileResult(
        source=graph,
        path_length=length,
        rows=rows,
    )


def _count_paths_by_endpoint(
    source: str,
    length: int,
    adj: dict[str, set[str]],
) -> dict[str, int]:
    """Count simple paths from one source, grouped by their endpoint."""
    counts: dict[str, int] = {}

    def visit(current: str, steps_left: int, visited: set[str]) -> None:
        if steps_left == 0:
            counts[current] = counts.get(current, 0) + 1
            return
        next_visited = visited | {current}
        for neighbor in adj[current]:
            if neighbor not in next_visited:
                visit(neighbor, steps_left - 1, next_visited)

    visit(source, length, set())
    return counts


__all__ = ["compute_path_profile"]
