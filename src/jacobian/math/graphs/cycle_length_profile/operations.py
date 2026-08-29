"""Cycle-length profile kernel."""

from __future__ import annotations

from jacobian.math.graphs.cycle_length_profile._models import (
    CycleLengthProfileResult,
    CycleLengthRow,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_cycle_length_profile"]


def compute_cycle_length_profile(
    graph: SimpleUndirectedGraph,
) -> CycleLengthProfileResult:
    """Return the complete cycle-length profile of a simple graph.

    For each length k from 3 to |V|, check if the graph contains a simple
    k-cycle. Return one canonical witness cycle for each present length.
    """
    vertices = list(graph.vertices)
    adjacency: dict[str, set[str]] = {v: set() for v in vertices}
    for a, b in graph.edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    n = len(vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    adj_matrix = [[False] * n for _ in range(n)]
    for a, b in graph.edges:
        i, j = vertex_to_idx[a], vertex_to_idx[b]
        adj_matrix[i][j] = True
        adj_matrix[j][i] = True

    found: dict[int, tuple[str, ...]] = {}
    for length in range(3, n + 1):
        witness = _find_cycle_of_length(length, n, adj_matrix, vertices)
        if witness is not None:
            found[length] = witness

    rows = [CycleLengthRow(cycle_length=k, witness=w) for k, w in sorted(found.items())]
    return CycleLengthProfileResult(graph=graph, rows=tuple(rows))


def _find_cycle_of_length(
    length: int,
    n: int,
    adj_matrix: list[list[bool]],
    vertices: list[str],
) -> tuple[str, ...] | None:
    """Find one simple cycle of the given length using DFS backtracking."""

    def dfs(
        start: int,
        current: int,
        visited: list[int],
        path: list[int],
    ) -> tuple[str, ...] | None:
        if len(path) == length:
            if adj_matrix[current][start]:
                return tuple(vertices[i] for i in path)
            return None
        for nxt in range(current + 1, n):
            if nxt not in visited and adj_matrix[current][nxt]:
                visited.append(nxt)
                result = dfs(start, nxt, visited, [*path, nxt])
                if result is not None:
                    return result
                visited.pop()
        return None

    for start in range(n):
        result = dfs(start, start, [start], [start])
        if result is not None:
            canonical = _canonicalize_cycle(result)
            return canonical
    return None


def _canonicalize_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    """Return the lexicographically smallest rotation of the cycle."""
    n = len(cycle)
    best = cycle
    for i in range(n):
        rotation = cycle[i:] + cycle[:i]
        if rotation < best:
            best = rotation
    return best
