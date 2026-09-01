"""Admission planning for exact open neighbourhoods."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)


@dataclass(frozen=True, slots=True)
class OpenNeighborhoodAdmission:
    """The canonical selected axis and exact result plan for one request."""

    selected_vertices: tuple[str, ...]
    neighborhood: tuple[str, ...]


def admit_open_neighborhood(
    graph: SimpleUndirectedGraph,
    selected_vertices: tuple[str, ...],
) -> OpenNeighborhoodAdmission:
    """Normalize one selected set and admit its exact mathematical result."""

    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError("open_neighborhood expects a SimpleUndirectedGraph")
    if not isinstance(selected_vertices, tuple) or any(
        not isinstance(vertex, str) for vertex in selected_vertices
    ):
        raise TypeError("selected_vertices must be a tuple of strings")
    if len(selected_vertices) > MAX_INDEXED_SIMPLE_GRAPH_VERTICES:
        raise OperationDomainValidationError(
            location=("selected_vertices",),
            code="graph.open_neighborhood.selected_vertices_bound",
            message=(
                "open-neighbourhood selection supports at most "
                f"{MAX_INDEXED_SIMPLE_GRAPH_VERTICES} raw vertices"
            ),
        )

    selected_set = set(selected_vertices)
    unknown = selected_set.difference(graph.vertices)
    if unknown:
        raise OperationDomainValidationError(
            location=("selected_vertices",),
            code="graph.open_neighborhood.selected_vertex_not_in_graph",
            message="every selected vertex must be a declared graph vertex",
        )

    selected = tuple(vertex for vertex in graph.vertices if vertex in selected_set)
    neighbors: set[str] = set()
    for left, right in graph.edges:
        if left in selected_set and right not in selected_set:
            neighbors.add(right)
        elif right in selected_set and left not in selected_set:
            neighbors.add(left)
    neighborhood = tuple(vertex for vertex in graph.vertices if vertex in neighbors)

    return OpenNeighborhoodAdmission(
        selected_vertices=selected,
        neighborhood=neighborhood,
    )


__all__ = [
    "OpenNeighborhoodAdmission",
    "admit_open_neighborhood",
]
