"""Typed contracts for the edge pattern profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


class EdgePatternProfileRequest(StrictModel):
    """Request for the vertex-colour edge-pattern profile of a hypergraph."""

    hypergraph: FiniteHypergraph
    vertex_colors: dict[str, str] = Field(
        description=(
            "A total map on the hypergraph vertex labels. The complete result "
            "is admitted against the canonical output-size envelope; color "
            "labels must be valid UTF-8 and are bounded by the aggregate "
            "output envelope."
        )
    )

    @model_validator(mode="after")
    def validate_colors(self) -> Self:
        vertices = set(self.hypergraph.vertices)
        if set(self.vertex_colors.keys()) != vertices:
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


class VertexColorPair(StrictModel):
    """One vertex-color pair in an edge-pattern profile coloring."""

    vertex: str
    color: str


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
