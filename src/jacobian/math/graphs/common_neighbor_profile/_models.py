"""Typed contracts for the common-neighbour profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 256


class CommonNeighborProfileRequest(StrictModel):
    """Request for the common-neighbour profile of a graph."""

    graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        if len(self.graph.vertices) > MAX_VERTICES:
            raise PydanticCustomError(
                "common_neighbor.too_many_vertices",
                f"at most {MAX_VERTICES} vertices are supported",
            )
        return self


class CommonNeighborRow(StrictModel):
    """One unordered vertex pair with its common-neighbour set."""

    vertex_u: str
    vertex_v: str
    common_neighbors: tuple[str, ...]
    codegree: int


class CommonNeighborProfileResult(StrictModel):
    """The complete common-neighbour profile of a graph."""

    graph: SimpleUndirectedGraph
    rows: tuple[CommonNeighborRow, ...]


__all__ = [
    "MAX_VERTICES",
    "CommonNeighborProfileRequest",
    "CommonNeighborProfileResult",
    "CommonNeighborRow",
]
