"""Exact open-neighbourhood kernel for a selected vertex set."""

from __future__ import annotations

from jacobian.math.graphs.neighborhood._bounds import (
    OpenNeighborhoodAdmission,
    admit_open_neighborhood,
)
from jacobian.math.graphs.neighborhood._models import NeighborhoodResult
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["open_neighborhood", "verify_open_neighborhood"]


def open_neighborhood(
    graph: SimpleUndirectedGraph,
    selected_vertices: tuple[str, ...],
) -> NeighborhoodResult:
    """Return the exact open neighbourhood N_G(S).

    The open neighbourhood of a vertex set S consists of all vertices
    outside S that are adjacent to at least one member of S.
    """
    return _open_neighborhood_from_admission(
        graph, admit_open_neighborhood(graph, selected_vertices)
    )


def _open_neighborhood_from_admission(
    graph: SimpleUndirectedGraph,
    admission: OpenNeighborhoodAdmission,
) -> NeighborhoodResult:
    """Compute the admitted neighbourhood and bind it to its source graph."""
    selected = set(admission.selected_vertices)
    neighbors: set[str] = set()
    for left, right in graph.edges:
        if left in selected and right not in selected:
            neighbors.add(right)
        elif right in selected and left not in selected:
            neighbors.add(left)
    return NeighborhoodResult(
        graph=graph,
        selected_vertices=admission.selected_vertices,
        neighborhood=tuple(vertex for vertex in graph.vertices if vertex in neighbors),
    )


def verify_open_neighborhood(claim: NeighborhoodResult) -> bool:
    """Return whether a serialized neighbourhood claim satisfies N_G(S)."""
    selected = set(claim.selected_vertices)
    neighbors: set[str] = set()
    for left, right in claim.graph.edges:
        if left in selected and right not in selected:
            neighbors.add(right)
        elif right in selected and left not in selected:
            neighbors.add(left)
    expected = tuple(vertex for vertex in claim.graph.vertices if vertex in neighbors)
    return claim.neighborhood == expected
