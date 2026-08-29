"""Exact open-neighbourhood kernel for a selected vertex set."""

from __future__ import annotations

from jacobian.math.graphs.neighborhood._models import (
    NeighborhoodResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["open_neighborhood"]


def open_neighborhood(
    graph: SimpleUndirectedGraph,
    selected: tuple[str, ...],
) -> NeighborhoodResult:
    """Return the exact open neighbourhood N_G(S).

    The open neighbourhood of a vertex set S consists of all vertices
    outside S that are adjacent to at least one member of S.
    """
    selected_set = set(selected)
    neighbors: set[str] = set()
    for left, right in graph.edges:
        if left in selected_set and right not in selected_set:
            neighbors.add(right)
        if right in selected_set and left not in selected_set:
            neighbors.add(left)
    vertex_order = {v: i for i, v in enumerate(graph.vertices)}
    sorted_neighbors = sorted(neighbors, key=lambda v: vertex_order[v])
    return NeighborhoodResult(
        graph=graph,
        selected_vertices=selected,
        neighborhood=tuple(sorted_neighbors),
    )
