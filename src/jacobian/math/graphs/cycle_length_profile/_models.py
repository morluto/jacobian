"""Typed contracts for the cycle-length profile operation."""

import math
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_CYCLE_LENGTH_SEARCH_WORK = 5_000_000


def _has_bounded_cycle_structure(graph: SimpleUndirectedGraph) -> bool:
    """Return whether path enumeration is polynomial for this source graph."""
    parent = {vertex: vertex for vertex in graph.vertices}
    degrees = dict.fromkeys(graph.vertices, 0)

    def find(vertex: str) -> str:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    forest = True
    for left, right in graph.edges:
        degrees[left] += 1
        degrees[right] += 1
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            forest = False
        else:
            parent[left_root] = right_root
    return forest or max(degrees.values(), default=0) <= 2


def _cycle_search_work(graph: SimpleUndirectedGraph) -> int:
    """Conservatively bound all simple paths explored across target lengths."""
    vertex_count = len(graph.vertices)
    if _has_bounded_cycle_structure(graph):
        return vertex_count**3
    paths_per_source = sum(
        math.perm(vertex_count - 1, depth) for depth in range(vertex_count)
    )
    return vertex_count * vertex_count * paths_per_source


class CycleLengthProfileRequest(StrictModel):
    """Request the complete simple-cycle length profile of a graph."""

    graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def require_bounded_search(self) -> Self:
        if _cycle_search_work(self.graph) > MAX_CYCLE_LENGTH_SEARCH_WORK:
            raise PydanticCustomError(
                "cycle_length.search_work_exceeded",
                "cycle-length enumeration exceeds the admitted simple-path work bound",
            )
        return self


class CycleLengthEntry(StrictModel):
    """One cycle length and a witness cycle."""

    length: int
    witness: tuple[str, ...]


class CycleLengthProfileResult(StrictModel):
    """The complete cycle-length profile of a graph."""

    graph: SimpleUndirectedGraph
    entries: tuple[CycleLengthEntry, ...]
    cycle_lengths: tuple[int, ...]


__all__ = [
    "MAX_CYCLE_LENGTH_SEARCH_WORK",
    "CycleLengthEntry",
    "CycleLengthProfileRequest",
    "CycleLengthProfileResult",
    "_cycle_search_work",
]
