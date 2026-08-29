"""Typed contracts for the hypergraph vertex containment operation."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_VERTICES = 20


class HypergraphVertexContainmentRequest(StrictModel):
    """Request for the vertex-containment probability profile of a hypergraph."""

    hypergraph: FiniteHypergraph
    retention_probability: CanonicalRational


class HypergraphVertexContainmentResult(StrictModel):
    """The complete vertex-containment profile of a hypergraph."""

    hypergraph: FiniteHypergraph
    retention_probability: CanonicalRational
    containing_subset_counts: tuple[int, ...]
    total_state_count: int
    success_count: int
    probability: CanonicalRational


__all__ = [
    "MAX_VERTICES",
    "HypergraphVertexContainmentRequest",
    "HypergraphVertexContainmentResult",
]
