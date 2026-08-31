"""Typed contracts for the edge pattern profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


class VertexColorPair(StrictModel):
    """One (vertex label, color label) pair in a vertex colouring.

    An aligned finite sequence of these pairs is the domain-owned carrier for
    a total vertex colouring: it is deterministic, order-insensitive for
    meaning, and cannot be mistaken for the repository's rational encoding.
    """

    vertex: str
    color: str


class EdgePatternProfileRequest(StrictModel):
    """Request for the vertex-colour edge-pattern profile of a hypergraph."""

    hypergraph: FiniteHypergraph
    vertex_colors: tuple[VertexColorPair, ...] = Field(
        description=(
            "A total finite vertex colouring as aligned (vertex, color) pairs "
            "covering every declared vertex exactly once. Color labels must be "
            "valid UTF-8 and are admitted against the aggregate output-size "
            "envelope."
        )
    )

    @model_validator(mode="after")
    def validate_colors(self) -> Self:
        vertices = set(self.hypergraph.vertices)
        rows = self.vertex_colors
        if len({pair.vertex for pair in rows}) != len(rows):
            raise PydanticCustomError(
                "edge_pattern.color_map_duplicate_vertex",
                "vertex_colors must not repeat a vertex",
            )
        if {pair.vertex for pair in rows} != vertices:
            raise PydanticCustomError(
                "edge_pattern.color_map_must_cover_all_vertices",
                "vertex_colors must cover exactly all declared vertices",
            )
        return self


class EdgePatternEntry(StrictModel):
    """One source edge with its colour equality pattern."""

    edge_id: str
    members: tuple[str, ...]
    equality_partition: tuple[int, ...]
    num_color_blocks: int
    color_labels: tuple[str, ...]


class EdgePatternProfileResult(StrictModel):
    """The complete vertex-colour edge-pattern profile of a hypergraph."""

    hypergraph: FiniteHypergraph
    vertex_colors: tuple[VertexColorPair, ...]
    entries: tuple[EdgePatternEntry, ...]
    monochromatic_edge_ids: tuple[str, ...]
    rainbow_edge_ids: tuple[str, ...]


__all__ = [
    "EdgePatternEntry",
    "EdgePatternProfileRequest",
    "EdgePatternProfileResult",
    "VertexColorPair",
]
