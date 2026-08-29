"""Open neighbourhood kernel."""

from __future__ import annotations

from jacobian.math.graphs.neighborhood.open_neighborhood._models import (
    OpenNeighborhoodResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_open_neighborhood"]


def compute_open_neighborhood(
    graph: SimpleUndirectedGraph,
    selected_vertices: tuple[str, ...],
) -> OpenNeighborhoodResult:
    """Return the exact sorted open neighbourhood of *selected_vertices*.

    The open neighbourhood N_G(S) consists of all graph vertices not in S
    that are adjacent to at least one member of S.
    """
    selected_set = frozenset(selected_vertices)
    neighborhood: set[str] = set()

    for left, right in graph.edges:
        if left in selected_set and right not in selected_set:
            neighborhood.add(right)
        if right in selected_set and left not in selected_set:
            neighborhood.add(left)

    return OpenNeighborhoodResult(
        graph=graph,
        selected_vertices=selected_vertices,
        neighborhood=tuple(sorted(neighborhood)),
    )
