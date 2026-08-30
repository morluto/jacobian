"""Typed contracts for the edge-deletion chromatic profile operation."""

import math
from collections import deque
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_EDGE_DELETION_COLORING_WORK = 10_000_000


def _is_bipartite(graph: SimpleUndirectedGraph) -> bool:
    adjacency: dict[str, list[str]] = {vertex: [] for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    colors: dict[str, int] = {}
    for source in graph.vertices:
        if source in colors:
            continue
        colors[source] = 0
        queue = deque([source])
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[vertex]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    return False
    return True


def _edge_deletion_admission_error(
    graph: SimpleUndirectedGraph, deletion_order: int
) -> tuple[str, str] | None:
    if deletion_order < 0:
        return ("nonnegative_order", "deletion order must be nonnegative")
    edge_count = len(graph.edges)
    maximum_order = min(deletion_order, edge_count)
    rows = sum(math.comb(edge_count, size) for size in range(maximum_order + 1))
    deleted_incidences = sum(
        size * math.comb(edge_count, size) for size in range(maximum_order + 1)
    )
    max_edge_bytes = (
        max((len(encode_strict_json([left, right])) + 8) for left, right in graph.edges)
        if graph.edges
        else 0
    )
    estimated_bytes = (
        len(encode_strict_json(graph.model_dump(mode="json")))
        + rows * 72
        + deleted_incidences * max_edge_bytes
        + 512
    )
    if estimated_bytes > CanonicalLimits().max_output_bytes:
        return ("output_bound", "edge-deletion profile exceeds the output budget")
    if not _is_bipartite(graph):
        vertex_count = len(graph.vertices)
        coloring_work = rows * (vertex_count + 1) * vertex_count**vertex_count
        if coloring_work > MAX_EDGE_DELETION_COLORING_WORK:
            return (
                "coloring_work_bound",
                "edge-deletion profile exceeds the exact-coloring work bound",
            )
    return None


class EdgeDeletionProfileRequest(StrictModel):
    """Request the edge-deletion chromatic profile of a graph."""

    graph: SimpleUndirectedGraph
    deletion_order: int

    @model_validator(mode="after")
    def require_bounded_profile(self) -> Self:
        failure = _edge_deletion_admission_error(self.graph, self.deletion_order)
        if failure is not None:
            code, message = failure
            raise PydanticCustomError(f"edge_deletion.{code}", message)
        return self


class DeletionEntry(StrictModel):
    """One deleted edge set and the resulting chromatic number."""

    deleted_edges: tuple[tuple[str, str], ...]
    chromatic_number: int


class EdgeDeletionProfileResult(StrictModel):
    """The complete edge-deletion chromatic profile."""

    graph: SimpleUndirectedGraph
    source_chromatic_number: int
    deletion_order: int
    entries: tuple[DeletionEntry, ...]


__all__ = [
    "MAX_EDGE_DELETION_COLORING_WORK",
    "DeletionEntry",
    "EdgeDeletionProfileRequest",
    "EdgeDeletionProfileResult",
    "_edge_deletion_admission_error",
    "_is_bipartite",
]
