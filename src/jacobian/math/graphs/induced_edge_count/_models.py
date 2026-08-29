"""Typed contracts for the induced-edge-count profile operation."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 256


class InducedEdgeCountProfileRequest(StrictModel):
    """Request for the induced-edge-count distribution at a fixed cardinality."""

    graph: SimpleUndirectedGraph
    cardinality: StrictInt = Field(ge=0, le=MAX_VERTICES)


class InducedEdgeCountRow(StrictModel):
    """One row of the induced-edge-count distribution."""

    edge_count: StrictInt = Field(ge=0)
    subset_count: StrictInt = Field(ge=1)
    witness: tuple[str, ...]


class InducedEdgeCountProfileResult(StrictModel):
    """The complete distribution of induced-edge counts over k-subsets."""

    graph: SimpleUndirectedGraph
    cardinality: StrictInt = Field(ge=0, le=MAX_VERTICES)
    rows: tuple[InducedEdgeCountRow, ...]


__all__ = [
    "MAX_VERTICES",
    "InducedEdgeCountProfileRequest",
    "InducedEdgeCountProfileResult",
    "InducedEdgeCountRow",
]
