"""Typed contracts for the hypergraph colouring decision operation."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


class HypergraphColoringRequest(StrictModel):
    """Request to decide q-colourability of a hypergraph."""

    hypergraph: FiniteHypergraph
    palette_size: int


class HypergraphColoringResult(StrictModel):
    """The hypergraph colouring decision."""

    hypergraph: FiniteHypergraph
    palette_size: int
    colorable: bool
    coloring: tuple[int, ...] | None = None


__all__ = [
    "HypergraphColoringRequest",
    "HypergraphColoringResult",
]
