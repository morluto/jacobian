"""Edge deletion profile kernel using brute-force chromatic number."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.graphs.edge_deletion_profile._models import (
    DeletionRow,
    EdgeDeletionProfileResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_edge_deletion_profile"]


def compute_edge_deletion_profile(
    graph: SimpleUndirectedGraph,
    deletion_order: int,
) -> EdgeDeletionProfileResult:
    """Return the chromatic number of G-F for every edge-deletion set F.

    For each subset F of edges with |F| <= deletion_order, compute the
    chromatic number of the graph after deleting those edges.
    """
    edges = list(graph.edges)
    vertices = list(graph.vertices)

    rows: list[DeletionRow] = []
    for order in range(deletion_order + 1):
        for edge_indices in combinations(range(len(edges)), order):
            remaining_edges = [
                edges[i] for i in range(len(edges)) if i not in set(edge_indices)
            ]
            chromatic = _chromatic_number(vertices, remaining_edges)
            rows.append(
                DeletionRow(
                    deleted_edge_indices=tuple(edge_indices),
                    chromatic_number=chromatic,
                )
            )

    return EdgeDeletionProfileResult(
        graph=graph,
        deletion_order=deletion_order,
        rows=tuple(rows),
    )


def _chromatic_number(vertices: list[str], edges: list[tuple[str, str]]) -> int:
    """Compute the exact chromatic number by brute-force search."""
    n = len(vertices)
    if n == 0:
        return 0
    if not edges:
        return 1

    adjacency: dict[str, set[str]] = {v: set() for v in vertices}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    for k in range(1, n + 1):
        if _try_k_color(vertices, adjacency, k):
            return k
    return n


def _try_k_color(vertices: list[str], adjacency: dict[str, set[str]], k: int) -> bool:
    """Check if the graph is k-colorable."""
    colors: dict[str, int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(vertices):
            return True
        v = vertices[idx]
        for c in range(k):
            if all(colors.get(n, -1) != c for n in adjacency[v]):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                del colors[v]
        return False

    return backtrack(0)
