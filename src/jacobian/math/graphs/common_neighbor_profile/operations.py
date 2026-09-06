"""Common-neighbour profile kernel."""

from __future__ import annotations

from typing import NoReturn

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.common_neighbor_profile._models import (
    MAX_VERTICES,
    CommonNeighborProfileResult,
    CommonNeighborRow,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_common_neighbor_profile", "verify_common_neighbor_profile"]

MAX_COMMON_NEIGHBOR_CELLS = 5_000_000
MAX_COMMON_NEIGHBOR_LABEL_CHARACTERS = 20_000_000


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("graph",), code=f"common_neighbor.{code}", message=message
    )


def _admit_graph(
    graph: SimpleUndirectedGraph,
) -> None:
    if not isinstance(graph, SimpleUndirectedGraph):
        _reject("invalid_graph", "graph must be a simple undirected graph")
    vertices = list(graph.vertices)
    if len(vertices) > MAX_VERTICES:
        _reject("too_many_vertices", f"at most {MAX_VERTICES} vertices are supported")

    degrees = dict.fromkeys(vertices, 0)
    for left, right in graph.edges:
        degrees[left] += 1
        degrees[right] += 1

    common_neighbor_cells = sum(
        degree * (degree - 1) // 2 for degree in degrees.values()
    )
    retained_label_characters = sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    retained_label_characters += (len(vertices) - 1) * sum(
        len(vertex) for vertex in vertices
    )
    retained_label_characters += sum(
        len(vertex) * degree * (degree - 1) // 2 for vertex, degree in degrees.items()
    )
    if common_neighbor_cells > MAX_COMMON_NEIGHBOR_CELLS:
        _reject(
            "result_cells_exceeded",
            "complete profile exceeds the "
            f"{MAX_COMMON_NEIGHBOR_CELLS:,}-common-neighbor-cell result bound",
        )
    if retained_label_characters > MAX_COMMON_NEIGHBOR_LABEL_CHARACTERS:
        _reject(
            "retained_labels_exceeded",
            "complete profile exceeds the retained label-character bound",
        )


def compute_common_neighbor_profile(
    graph: SimpleUndirectedGraph,
) -> CommonNeighborProfileResult:
    """Return the complete common-neighbour profile of a simple graph.

    For every unordered pair of distinct vertices, return the sorted
    set of common neighbours, its cardinality (codegree), in canonical
    source-vertex order.
    """
    _admit_graph(graph)
    rows = _common_neighbor_rows(graph)
    return CommonNeighborProfileResult(graph=graph, rows=tuple(rows))


def verify_common_neighbor_profile(claim: CommonNeighborProfileResult) -> bool:
    """Return whether every claimed row is the complete profile of its graph."""
    return claim.rows == tuple(_common_neighbor_rows(claim.graph))


def _common_neighbor_rows(graph: SimpleUndirectedGraph) -> list[CommonNeighborRow]:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    rows: list[CommonNeighborRow] = []
    for index, left in enumerate(graph.vertices):
        for right in graph.vertices[index + 1 :]:
            common = tuple(sorted(adjacency[left] & adjacency[right]))
            rows.append(
                CommonNeighborRow(
                    vertex_u=left,
                    vertex_v=right,
                    common_neighbors=common,
                    codegree=len(common),
                )
            )
    return rows
