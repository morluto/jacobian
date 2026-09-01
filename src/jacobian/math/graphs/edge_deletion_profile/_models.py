"""Typed contracts for the edge deletion profile operation."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, WithJsonSchema
from pydantic.json_schema import JsonSchemaValue

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_EDGES,
    SimpleUndirectedGraph,
)

MAX_DELETION_ORDER = MAX_INDEXED_SIMPLE_GRAPH_EDGES


def _edge_deletion_graph_schema() -> JsonSchemaValue:
    """Describe the shared graph axis and the operation's work-sensitive bound."""

    schema = SimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "A finite simple graph on Jacobian's canonical graph axis. The profile "
        "operation admits requests by retained label characters, deletion-row "
        "count, and coloring work."
    )
    return schema


EdgeDeletionGraph = Annotated[
    SimpleUndirectedGraph,
    WithJsonSchema(_edge_deletion_graph_schema()),
]


class EdgeDeletionProfileRequest(StrictModel):
    """Request for the edge deletion chromatic profile of a graph."""

    graph: EdgeDeletionGraph
    deletion_order: int = Field(
        ge=0,
        le=MAX_DELETION_ORDER,
        strict=True,
        description=(
            "Maximum number of deleted edges; it must not exceed the graph's "
            "edge count."
        ),
    )


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
    "DeletionRow",
    "EdgeDeletionGraph",
    "EdgeDeletionProfileRequest",
    "EdgeDeletionProfileResult",
]
