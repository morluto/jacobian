"""Common-neighbour profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.common_neighbor_profile._models import (
    MAX_VERTICES,
    CommonNeighborProfileResult,
    CommonNeighborRow,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_common_neighbor_profile"]

MAX_COMMON_NEIGHBOR_CELLS = 5_000_000
MAX_COMMON_NEIGHBOR_LABEL_CHARACTERS = 20_000_000


@dataclass(frozen=True, slots=True)
class _ProfilePlan:
    rows: tuple[tuple[str, str, tuple[str, ...]], ...]


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("graph",), code=f"common_neighbor.{code}", message=message
    )


def _admit_graph(
    graph: SimpleUndirectedGraph,
) -> tuple[dict[str, set[str]], _ProfilePlan]:
    if not isinstance(graph, SimpleUndirectedGraph):
        _reject("invalid_graph", "graph must be a simple undirected graph")
    vertices = list(graph.vertices)
    if len(vertices) > MAX_VERTICES:
        _reject("too_many_vertices", f"at most {MAX_VERTICES} vertices are supported")

    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    rows: list[tuple[str, str, tuple[str, ...]]] = []
    common_neighbor_cells = 0
    retained_label_characters = sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    for index, left in enumerate(vertices):
        for right in vertices[index + 1 :]:
            common = tuple(sorted(adjacency[left] & adjacency[right]))
            common_neighbor_cells += len(common)
            retained_label_characters += (
                len(left) + len(right) + sum(len(vertex) for vertex in common)
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
            rows.append((left, right, common))

    return adjacency, _ProfilePlan(rows=tuple(rows))


def compute_common_neighbor_profile(
    graph: SimpleUndirectedGraph,
) -> CommonNeighborProfileResult:
    """Return the complete common-neighbour profile of a simple graph.

    For every unordered pair of distinct vertices, return the sorted
    set of common neighbours, its cardinality (codegree), in canonical
    source-vertex order.
    """
    _adjacency, plan = _admit_graph(graph)
    rows = [
        CommonNeighborRow(
            vertex_u=left,
            vertex_v=right,
            common_neighbors=common,
            codegree=len(common),
        )
        for left, right, common in plan.rows
    ]
    return CommonNeighborProfileResult(graph=graph, rows=tuple(rows))
