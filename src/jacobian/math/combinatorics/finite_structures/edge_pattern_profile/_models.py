"""Typed contracts for the edge pattern profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_LABEL_LENGTH,
    FiniteHypergraph,
)


class EdgePatternProfileRequest(StrictModel):
    """Request for the vertex-colour edge-pattern profile of a hypergraph."""

    hypergraph: FiniteHypergraph
    vertex_colors: dict[str, str] = Field(
        description=(
            "A total map on the hypergraph vertex labels. The complete result "
            "retains one bounded color pattern per source edge; each color label "
            f"has at most {MAX_LABEL_LENGTH} characters."
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


class VertexColoring(StrictModel):
    """A total coloring bound to the source hypergraph vertex axis."""

    hypergraph: FiniteHypergraph
    assignments: tuple[VertexColorPair, ...]

    @model_validator(mode="after")
    def require_source_axis_shape(self) -> Self:
        vertices = tuple(self.hypergraph.vertices)
        assigned = tuple(pair.vertex for pair in self.assignments)
        if assigned != vertices:
            raise PydanticCustomError(
                "edge_pattern.coloring_axis",
                "coloring assignments must follow the complete source vertex axis",
            )
        return self


class EdgePatternProfileResult(StrictModel):
    """The complete vertex-colour edge-pattern profile of a hypergraph."""

    hypergraph: FiniteHypergraph
    vertex_coloring: VertexColoring
    entries: tuple[EdgePatternEntry, ...]
    monochromatic_edge_ids: tuple[str, ...]
    rainbow_edge_ids: tuple[str, ...]


__all__ = [
    "EdgePatternEntry",
    "EdgePatternProfileRequest",
    "EdgePatternProfileResult",
    "VertexColorPair",
    "VertexColoring",
]
