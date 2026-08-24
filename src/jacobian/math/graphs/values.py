"""Provider-independent values for finite simple undirected graphs."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from jacobian._models import StrictModel

GraphCompositionOperation = Literal[
    "DISJOINT_UNION",
    "JOIN",
    "COMPLEMENT",
    "LEXICOGRAPHIC_PRODUCT",
]

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


class SimpleUndirectedGraph(StrictModel):
    """Immutable canonical value for a finite simple undirected graph."""

    graph_schema_version: Literal["1"] = "1"
    vertices: tuple[str, ...] = Field(max_length=256)
    edges: tuple[tuple[str, str], ...] = Field(max_length=32640)

    @model_validator(mode="after")
    def require_canonical_simple_graph(self) -> Self:
        if any(
            not unicodedata.is_normalized("NFC", vertex) for vertex in self.vertices
        ):
            raise ValueError("graph vertices must use Unicode NFC")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("graph vertices must be unique")
        if any(
            left >= right or left not in self.vertices or right not in self.vertices
            for left, right in self.edges
        ):
            raise ValueError("edges must contain two declared vertices in order")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("graph edges must be unique")
        return self


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
            "The domain-owned canonical simple undirected graph. Each vertex "
            "label is preserved exactly, must use Unicode NFC, and may contain "
            "at most 64 UTF-8 bytes."
        ),
    )
    vertex_colors: tuple[GraphColor, ...] = Field(
        default=(),
        description=(
            "Either empty for an uncolored vertex set or one color per vertex, "
            "aligned with `graph.vertices`. Color names are preserved exactly; "
            "each name must use Unicode NFC and at most 64 UTF-8 bytes (the "
            "schema `maxLength` counts characters, not UTF-8 bytes)."
        ),
    )
    edge_colors: tuple[GraphColor, ...] = Field(
        default=(),
        description=(
            "Either empty for an uncolored edge set or one color per edge, "
            "aligned with `graph.edges`. Color names are preserved exactly; "
            "each name must use Unicode NFC and at most 64 UTF-8 bytes (the "
            "schema `maxLength` counts characters, not UTF-8 bytes)."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_colored_graph(self) -> Self:
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


class GraphCompositionInput(StrictModel):
    """Two exact graphs and one explicit graph composition operation."""

    operation: GraphCompositionOperation
    left: SimpleUndirectedGraph
    right: SimpleUndirectedGraph | None = None

    @model_validator(mode="after")
    def require_operands(self) -> Self:
        if self.operation == "COMPLEMENT":
            if self.right is not None:
                raise ValueError("complement does not accept a right graph")
        elif self.right is None:
            raise ValueError(f"{self.operation} requires a right graph")
        return self


__all__ = [
    "MAX_GRAPH_COLOR_BYTES",
    "MAX_GRAPH_LABEL_BYTES",
    "ColoredUndirectedGraph",
    "GraphColor",
    "GraphCompositionInput",
    "GraphCompositionOperation",
    "GraphVertexLabel",
    "SimpleUndirectedGraph",
]
