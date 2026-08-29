"""Typed contracts for the edge deletion profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 8
MAX_EDGES = 12
MAX_DELETION_ORDER = 3


class EdgeDeletionProfileRequest(StrictModel):
    """Request for the edge deletion chromatic profile of a graph."""

    graph: SimpleUndirectedGraph
    deletion_order: int = Field(ge=0, le=MAX_DELETION_ORDER)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if len(self.graph.vertices) > MAX_VERTICES:
            raise PydanticCustomError(
                "edge_deletion.too_many_vertices",
                f"at most {MAX_VERTICES} vertices are supported",
            )
        if len(self.graph.edges) > MAX_EDGES:
            raise PydanticCustomError(
                "edge_deletion.too_many_edges",
                f"at most {MAX_EDGES} edges are supported",
            )
        if self.deletion_order > len(self.graph.edges):
            raise PydanticCustomError(
                "edge_deletion.order_exceeds_edge_count",
                "deletion_order must not exceed the number of edges",
            )
        return self


class DeletionRow(StrictModel):
    """One edge-deletion subset and its chromatic number."""

    deleted_edge_indices: tuple[int, ...]
    chromatic_number: int


class EdgeDeletionProfileResult(StrictModel):
    """The complete edge deletion chromatic profile of a graph."""

    graph: SimpleUndirectedGraph
    deletion_order: int
    rows: tuple[DeletionRow, ...]


__all__ = [
    "MAX_DELETION_ORDER",
    "MAX_EDGES",
    "MAX_VERTICES",
    "DeletionRow",
    "EdgeDeletionProfileRequest",
    "EdgeDeletionProfileResult",
]
