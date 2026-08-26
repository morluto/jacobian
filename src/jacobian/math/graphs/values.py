"""Provider-independent values for finite simple undirected graphs."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json

GraphCompositionOperation = Literal[
    "DISJOINT_UNION",
    "JOIN",
    "COMPLEMENT",
    "LEXICOGRAPHIC_PRODUCT",
]

MAX_GRAPH_LABEL_BYTES = 64
MAX_GRAPH_COLOR_BYTES = 64
MAX_INDEXED_SIMPLE_GRAPH_VERTICES = 256
MAX_INDEXED_SIMPLE_GRAPH_EDGES = (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES * (MAX_INDEXED_SIMPLE_GRAPH_VERTICES - 1) // 2
)

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
        raise PydanticCustomError(
            "graph.kind_empty_if_unicodedata_normalized_nfc_value",
            f"{kind} must not be empty",
        )
    if not unicodedata.is_normalized("NFC", value):
        raise PydanticCustomError(
            "graph.kind_use_unicode_nfc_if_len_value", f"{kind} must use Unicode NFC"
        )
    if len(value.encode("utf-8")) > max_bytes:
        raise PydanticCustomError(
            "graph.kind_use_at_most_max_bytes_utf",
            f"{kind} must use at most {max_bytes} UTF-8 bytes",
        )


class SimpleUndirectedGraph(StrictModel):
    """Immutable canonical value for a finite simple undirected graph."""

    vertices: tuple[str, ...] = Field(max_length=256)
    edges: tuple[tuple[str, str], ...] = Field(max_length=32640)

    @model_validator(mode="after")
    def require_canonical_simple_graph(self) -> Self:
        if any(
            not unicodedata.is_normalized("NFC", vertex) for vertex in self.vertices
        ):
            raise PydanticCustomError(
                "graph.graph_vertices_must_use_unicode_nfc",
                "graph vertices must use Unicode NFC",
            )
        if len(set(self.vertices)) != len(self.vertices):
            raise PydanticCustomError(
                "graph.graph_vertices_must_be_unique", "graph vertices must be unique"
            )
        if any(
            left >= right or left not in self.vertices or right not in self.vertices
            for left, right in self.edges
        ):
            raise PydanticCustomError(
                "graph.edges_must_contain_two_declared_vertices_in_orde",
                "edges must contain two declared vertices in order",
            )
        if len(set(self.edges)) != len(self.edges):
            raise PydanticCustomError(
                "graph.graph_edges_must_be_unique", "graph edges must be unique"
            )
        return self


class IndexedSimpleUndirectedGraph(StrictModel):
    """A finite simple undirected graph on the integer axis ``0..n-1``.

    Edges are canonical ordered pairs ``(left, right)`` with
    ``left < right``, matching ``SimpleUndirectedGraph``, so serialized
    equality identifies mathematically identical graphs.  The null graph
    (``vertex_count=0`` with no edges) is a valid canonical value;
    operations admitting only nonempty graphs enforce that envelope in
    their own request validators.
    """

    vertex_count: int = Field(ge=0, le=MAX_INDEXED_SIMPLE_GRAPH_VERTICES)
    edges: tuple[tuple[int, int], ...] = Field(
        max_length=MAX_INDEXED_SIMPLE_GRAPH_EDGES
    )

    @model_validator(mode="after")
    def require_simple_indexed_graph(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for left, right in self.edges:
            if not (0 <= left < self.vertex_count and 0 <= right < self.vertex_count):
                raise PydanticCustomError(
                    "graph.edge_vertices_must_be_in_0_vertex_count_1",
                    "edge vertices must be in 0..vertex_count-1",
                )
            if left == right:
                raise PydanticCustomError(
                    "graph.a_simple_graph_cannot_contain_self_loops",
                    "a simple graph cannot contain self-loops",
                )
            if left > right:
                raise PydanticCustomError(
                    "graph.indexed_edges_must_be_canonical_pairs_with_left",
                    "indexed edges must be canonical pairs with left < right",
                )
            if (left, right) in seen:
                raise PydanticCustomError(
                    "graph.a_simple_graph_cannot_contain_duplicate_edges",
                    "a simple graph cannot contain duplicate edges",
                )
            seen.add((left, right))
        return self


def simple_undirected_graph_wire_bytes(graph: SimpleUndirectedGraph) -> int:
    """Return the exact canonical JSON size of a simple graph value."""

    return len(encode_strict_json(graph.model_dump(mode="json")))


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
            raise PydanticCustomError(
                "graph.vertex_colors_empty_align_one_color_with",
                "vertex_colors must be empty or align one color with every vertex",
            )
        if self.edge_colors and len(self.edge_colors) != len(self.graph.edges):
            raise PydanticCustomError(
                "graph.edge_colors_empty_align_one_color_with",
                "edge_colors must be empty or align one color with every edge",
            )
        for color in (*self.vertex_colors, *self.edge_colors):
            _require_canonical_text(
                color,
                kind="graph colors",
                max_bytes=MAX_GRAPH_COLOR_BYTES,
            )
        return self


__all__ = [
    "MAX_GRAPH_COLOR_BYTES",
    "MAX_GRAPH_LABEL_BYTES",
    "MAX_INDEXED_SIMPLE_GRAPH_EDGES",
    "MAX_INDEXED_SIMPLE_GRAPH_VERTICES",
    "ColoredUndirectedGraph",
    "GraphColor",
    "GraphCompositionOperation",
    "GraphVertexLabel",
    "IndexedSimpleUndirectedGraph",
    "SimpleUndirectedGraph",
    "simple_undirected_graph_wire_bytes",
]
