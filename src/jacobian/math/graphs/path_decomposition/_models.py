"""Typed contracts for the path decomposition operation."""

from __future__ import annotations

from itertools import pairwise
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 256


class PathDecompositionRequest(StrictModel):
    """Request for the minimum path decomposition of a graph."""

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Simple undirected graph with at most "
            f"{MAX_VERTICES} vertices; this operation's graph-sensitive path search "
            "also applies a graph-sensitive work bound."
        )
    )

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        if len(self.graph.vertices) > MAX_VERTICES:
            raise PydanticCustomError(
                "path_decomposition.too_many_vertices",
                f"at most {MAX_VERTICES} vertices are supported",
            )
        return self


class PathDecompositionResult(StrictModel):
    """The minimum path decomposition of a graph."""

    graph: SimpleUndirectedGraph
    path_count: int = Field(ge=0)
    paths: tuple[tuple[str, ...], ...]

    @model_validator(mode="after")
    def require_source_edge_partition(self) -> Self:
        if self.path_count != len(self.paths):
            raise PydanticCustomError(
                "path_decomposition.path_count",
                "path_count must equal the number of returned paths",
            )
        source_edges = set(self.graph.edges)
        used_edges: set[tuple[str, str]] = set()
        source_vertices = set(self.graph.vertices)
        for path in self.paths:
            if len(path) < 2 or len(path) != len(set(path)):
                raise PydanticCustomError(
                    "path_decomposition.simple_path",
                    "each returned path must contain distinct vertices and at least one edge",
                )
            if any(vertex not in source_vertices for vertex in path):
                raise PydanticCustomError(
                    "path_decomposition.unknown_vertex",
                    "returned paths must use vertices from the source graph",
                )
            for left, right in pairwise(path):
                edge = (left, right) if left < right else (right, left)
                if edge not in source_edges or edge in used_edges:
                    raise PydanticCustomError(
                        "path_decomposition.edge_partition",
                        "returned paths must partition the source edges exactly once",
                    )
                used_edges.add(edge)
        if used_edges != source_edges:
            raise PydanticCustomError(
                "path_decomposition.edge_partition",
                "returned paths must partition the source edges exactly once",
            )
        return self


__all__ = [
    "MAX_VERTICES",
    "PathDecompositionRequest",
    "PathDecompositionResult",
]
