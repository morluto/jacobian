"""Typed contracts for the monochromatic path hypergraph operation."""

import math
from collections import defaultdict
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph

MAX_MONOCHROMATIC_PATH_SEARCH_WORK = 2_000_000


def _component_sizes(
    vertices: tuple[str, ...], edges: list[tuple[str, str]]
) -> tuple[int, ...]:
    parent = {vertex: vertex for vertex in vertices}

    def find(vertex: str) -> str:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root
    sizes: dict[str, int] = defaultdict(int)
    for vertex in vertices:
        sizes[find(vertex)] += 1
    return tuple(sizes.values())


def _monochromatic_path_admission_error(
    graph: ColoredUndirectedGraph,
) -> tuple[str, str] | None:
    vertices = graph.graph.vertices
    color_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    colors = graph.edge_colors or tuple("uncolored" for _ in graph.graph.edges)
    for edge, color in zip(graph.graph.edges, colors, strict=True):
        color_edges[color].append(edge)

    total_supports = 0
    total_incidences = 0
    total_work = 0
    for edges in color_edges.values():
        supports = len(vertices)
        incidences = len(vertices)
        for size in _component_sizes(vertices, edges):
            supports += 2**size - size - 1
            incidences += size * 2 ** (size - 1) - size
            total_work += sum(math.perm(size, length) for length in range(2, size + 1))
        total_supports += supports
        total_incidences += incidences
    if total_supports > MAX_EDGES or total_incidences > MAX_TOTAL_INCIDENCES:
        return ("output_bound", "monochromatic path families exceed result bounds")
    if total_work > MAX_MONOCHROMATIC_PATH_SEARCH_WORK:
        return (
            "work_bound",
            "monochromatic path enumeration exceeds the admitted search-work bound",
        )
    return None


class MonochromaticPathRequest(StrictModel):
    """Request to construct monochromatic path candidate hypergraphs."""

    graph: ColoredUndirectedGraph

    @model_validator(mode="after")
    def require_bounded_path_families(self) -> Self:
        failure = _monochromatic_path_admission_error(self.graph)
        if failure is not None:
            code, message = failure
            raise PydanticCustomError(f"monochromatic_path.{code}", message)
        return self


class MonochromaticPathResult(StrictModel):
    """One per-colour monochromatic path hypergraph."""

    color: str
    hypergraph: FiniteHypergraph


class MonochromaticPathHypergraphResult(StrictModel):
    """Complete monochromatic path candidate hypergraph family."""

    graph: ColoredUndirectedGraph
    per_color: tuple[MonochromaticPathResult, ...]


__all__ = [
    "MAX_MONOCHROMATIC_PATH_SEARCH_WORK",
    "MonochromaticPathHypergraphResult",
    "MonochromaticPathRequest",
    "MonochromaticPathResult",
    "_monochromatic_path_admission_error",
]
