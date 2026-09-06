"""Typed contracts for the path decomposition operation."""

from __future__ import annotations

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
    """The minimum path decomposition of a graph.

    Parsing is structural only: path shape, vertex membership, and count
    alignment. Whether paths partition the source edges is a caller-authored
    claim checked by ``verify_path_decomposition``; minimum-count optimality
    is the producer's outcome, not a parsing invariant.
    """

    graph: SimpleUndirectedGraph
    path_count: int = Field(ge=0)
    paths: tuple[tuple[str, ...], ...]

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        if self.path_count != len(self.paths):
            raise PydanticCustomError(
                "path_decomposition.path_count",
                "path_count must equal the number of returned paths",
            )
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
        return self

    @classmethod
    def _from_kernel(
        cls,
        graph: SimpleUndirectedGraph,
        path_count: int,
        paths: tuple[tuple[str, ...], ...],
    ) -> Self:
        """Construct a result after the kernel established the partition."""

        return cls.model_construct(graph=graph, path_count=path_count, paths=paths)


__all__ = [
    "MAX_VERTICES",
    "PathDecompositionRequest",
    "PathDecompositionResult",
]
