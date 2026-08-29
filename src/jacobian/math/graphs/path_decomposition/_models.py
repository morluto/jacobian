"""Typed contracts for the path decomposition operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 12


class PathDecompositionRequest(StrictModel):
    """Request for the minimum path decomposition of a graph."""

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Simple undirected graph with at most "
            f"{MAX_VERTICES} vertices; this operation's exhaustive path search "
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
    path_count: int
    paths: tuple[tuple[str, ...], ...]


__all__ = [
    "MAX_VERTICES",
    "PathDecompositionRequest",
    "PathDecompositionResult",
]
