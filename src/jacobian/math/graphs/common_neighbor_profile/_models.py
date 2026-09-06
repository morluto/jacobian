"""Typed contracts for the common-neighbour profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
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
    codegree: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_canonical_common_neighbors(self) -> Self:
        if self.common_neighbors != tuple(sorted(set(self.common_neighbors))):
            raise PydanticCustomError(
                "common_neighbor.common_neighbors_must_be_sorted_unique",
                "common neighbours must be sorted and unique",
            )
        return self


class CommonNeighborProfileResult(StrictModel):
    """The complete common-neighbour profile of a graph."""

    graph: SimpleUndirectedGraph
    rows: tuple[CommonNeighborRow, ...] = Field(
        max_length=MAX_VERTICES * (MAX_VERTICES - 1) // 2
    )

    @classmethod
    def _from_kernel(
        cls,
        graph: SimpleUndirectedGraph,
        rows: tuple[CommonNeighborRow, ...],
    ) -> Self:
        """Construct a result after admission and the kernel relation."""

        return cls.model_construct(graph=graph, rows=rows)


__all__ = [
    "MAX_VERTICES",
    "CommonNeighborProfileRequest",
    "CommonNeighborProfileResult",
    "CommonNeighborRow",
]
