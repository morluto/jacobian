"""Typed contracts for the induced-edge-count profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 20


class InducedEdgeCountProfileRequest(StrictModel):
    """Request for the induced-edge-count distribution at a fixed cardinality."""

    graph: SimpleUndirectedGraph
    cardinality: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_cardinality(self) -> Self:
        if self.cardinality > len(self.graph.vertices):
            raise PydanticCustomError(
                "graph.induced_edge_count.cardinality_too_large",
                "cardinality must not exceed the number of graph vertices",
            )
        if len(self.graph.vertices) > MAX_VERTICES:
            raise PydanticCustomError(
                "graph.induced_edge_count.too_many_vertices",
                f"at most {MAX_VERTICES} vertices are supported",
            )
        return self


class InducedEdgeCountRow(StrictModel):
    """One row of the induced-edge-count distribution."""

    edge_count: int
    subset_count: int
    witness: tuple[str, ...]


class InducedEdgeCountProfileResult(StrictModel):
    """The complete distribution of induced-edge counts over k-subsets."""

    graph: SimpleUndirectedGraph
    cardinality: int
    rows: tuple[InducedEdgeCountRow, ...]


__all__ = [
    "MAX_VERTICES",
    "InducedEdgeCountProfileRequest",
    "InducedEdgeCountProfileResult",
    "InducedEdgeCountRow",
]
