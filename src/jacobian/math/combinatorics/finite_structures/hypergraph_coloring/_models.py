"""Typed contracts for hypergraph non-monochromatic colouring decision."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_PALETTE_SIZE = 16
MAX_VERTEX_COUNT = 16
MAX_EDGE_COUNT = 200

ColoringResult = str  # Literal["COLORABLE", "NOT_COLORABLE"]


class NonmonochromaticColoringRequest(StrictModel):
    """Decide whether a hypergraph has a q-colouring with no monochromatic edge."""

    hypergraph: FiniteHypergraph
    palette_size: int = Field(ge=1, le=MAX_PALETTE_SIZE)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if len(self.hypergraph.vertices) > MAX_VERTEX_COUNT:
            raise PydanticCustomError(
                "hypergraph_coloring.too_many_vertices",
                f"at most {MAX_VERTEX_COUNT} vertices are supported",
            )
        if len(self.hypergraph.edges) > MAX_EDGE_COUNT:
            raise PydanticCustomError(
                "hypergraph_coloring.too_many_edges",
                f"at most {MAX_EDGE_COUNT} edges are supported",
            )
        return self


class ColoringWitness(StrictModel):
    """A complete vertex-to-colour assignment."""

    assignments: tuple[tuple[str, int], ...]


class NonmonochromaticColoringResult(StrictModel):
    """Result of a non-monochromatic colouring decision."""

    hypergraph: FiniteHypergraph
    palette_size: int
    outcome: ColoringResult
    witness: ColoringWitness | None = None


__all__ = [
    "MAX_EDGE_COUNT",
    "MAX_PALETTE_SIZE",
    "MAX_VERTEX_COUNT",
    "ColoringWitness",
    "NonmonochromaticColoringRequest",
    "NonmonochromaticColoringResult",
]
