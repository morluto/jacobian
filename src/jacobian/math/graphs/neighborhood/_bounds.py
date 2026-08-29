"""Admission planning for exact open neighbourhoods."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
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


def _encoded_size(value: object, *, location: tuple[str, ...]) -> int:
    try:
        return len(encode_strict_json(value))
    except CanonicalizationError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="graph.open_neighborhood.output_budget",
            message=(
                "the open-neighbourhood result cannot fit the canonical output budget"
            ),
        ) from exc


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


def require_open_neighborhood_output_budget(
    graph: SimpleUndirectedGraph,
    selected_vertices: tuple[str, ...],
    neighborhood: tuple[str, ...],
) -> None:
    """Reject a result that cannot cross the canonical transport boundary."""

    graph_bytes = _encoded_size(
        graph.model_dump(mode="json"),
        location=("graph",),
    )
    selected_bytes = _encoded_size(
        list(selected_vertices),
        location=("selected_vertices",),
    )
    neighborhood_bytes = _encoded_size(
        list(neighborhood),
        location=("graph",),
    )
    output_bytes = strict_json_object_size(
        (
            ("graph", graph_bytes),
            ("selected_vertices", selected_bytes),
            ("neighborhood", neighborhood_bytes),
        )
    )
    output_limit = CanonicalLimits().max_output_bytes
    if output_bytes > output_limit:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.open_neighborhood.output_budget",
            message=(
                "the open-neighbourhood result retains its source graph, selected "
                f"axis, and exact neighbourhood ({output_bytes:,} bytes), exceeding "
                f"the {output_limit:,}-byte canonical output budget"
            ),
        )


__all__ = [
    "OpenNeighborhoodAdmission",
    "admit_open_neighborhood",
    "require_open_neighborhood_output_budget",
]
