"""Canonical values for finite vertex- and edge-colored graphs."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_COLORED_GRAPH_VERTICES = 64
MAX_COLORED_GRAPH_EDGES = (
    MAX_COLORED_GRAPH_VERTICES * (MAX_COLORED_GRAPH_VERTICES - 1) // 2
)
MAX_GRAPH_LABEL_BYTES = 64
MAX_GRAPH_COLOR_BYTES = 64

GraphVertexLabel = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=MAX_GRAPH_LABEL_BYTES),
]
GraphColor = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=MAX_GRAPH_COLOR_BYTES),
]


def _require_canonical_text(value: str, *, kind: str, max_bytes: int) -> None:
    if not value:
        raise ValueError(f"{kind} must not be empty")
    if not unicodedata.is_normalized("NFC", value):
        raise ValueError(f"{kind} must use Unicode NFC")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{kind} must use at most {max_bytes} UTF-8 bytes")


class ColoredUndirectedGraph(StrictModel):
    """One materialized finite simple graph with optional total colorings.

    ``vertex_colors[i]`` colors ``graph.vertices[i]`` and ``edge_colors[i]``
    colors ``graph.edges[i]``.  An empty color tuple means that the
    corresponding objects are uncolored; otherwise it must cover the complete
    authoritative axis of the domain-owned ``SimpleUndirectedGraph``.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A materialized canonical `SimpleUndirectedGraph` plus optional "
                "total colorings. Each color tuple is either empty (uncolored) "
                "or is aligned with the graph's authoritative vertex or edge "
                "axis."
            )
        }
    )

    colored_graph_schema_version: Literal["1"] = "1"
    graph: SimpleUndirectedGraph = Field(
        description=(
            "The domain-owned canonical simple undirected graph. This colored "
            "value admits at most 64 vertices, 2,016 edges, and 64 UTF-8 bytes "
            "per vertex label."
        ),
    )
    vertex_colors: tuple[GraphColor, ...] = Field(
        default=(),
        max_length=MAX_COLORED_GRAPH_VERTICES,
        description=(
            "Either empty for an uncolored vertex set or one color per vertex, "
            "aligned with `graph.vertices`. Color names are preserved exactly."
        ),
    )
    edge_colors: tuple[GraphColor, ...] = Field(
        default=(),
        max_length=MAX_COLORED_GRAPH_EDGES,
        description=(
            "Either empty for an uncolored edge set or one color per edge, "
            "aligned with `graph.edges`. Color names are preserved exactly."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_colored_graph(self) -> Self:
        if len(self.graph.vertices) > MAX_COLORED_GRAPH_VERTICES:
            raise ValueError(
                f"colored graphs support at most {MAX_COLORED_GRAPH_VERTICES} vertices"
            )
        if len(self.graph.edges) > MAX_COLORED_GRAPH_EDGES:
            raise ValueError(
                f"colored graphs support at most {MAX_COLORED_GRAPH_EDGES} edges"
            )
        for vertex in self.graph.vertices:
            _require_canonical_text(
                vertex,
                kind="graph vertex labels",
                max_bytes=MAX_GRAPH_LABEL_BYTES,
            )

        if self.vertex_colors and len(self.vertex_colors) != len(self.graph.vertices):
            raise ValueError(
                "vertex_colors must be empty or align one color with every vertex"
            )
        if self.edge_colors and len(self.edge_colors) != len(self.graph.edges):
            raise ValueError(
                "edge_colors must be empty or align one color with every edge"
            )
        for color in (*self.vertex_colors, *self.edge_colors):
            _require_canonical_text(
                color,
                kind="graph colors",
                max_bytes=MAX_GRAPH_COLOR_BYTES,
            )
        return self


__all__ = [
    "MAX_COLORED_GRAPH_EDGES",
    "MAX_COLORED_GRAPH_VERTICES",
    "MAX_GRAPH_COLOR_BYTES",
    "MAX_GRAPH_LABEL_BYTES",
    "ColoredUndirectedGraph",
    "GraphColor",
    "GraphVertexLabel",
]
