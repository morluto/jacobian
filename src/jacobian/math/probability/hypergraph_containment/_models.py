"""Typed contracts for the hypergraph vertex containment operation."""

from pydantic import Field

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_SUBSET_STATES = 1 << 22
MAX_CONTAINMENT_WORK = 20_000_000


class HypergraphVertexContainmentRequest(StrictModel):
    """Request for the vertex-containment probability profile of a hypergraph."""

    hypergraph: FiniteHypergraph = Field(
        description=(
            "Finite hypergraph. Complete subset enumeration is admitted by "
            "the derived state and edge-scan work envelope."
        )
    )
    retention_probability: CanonicalRational = Field(
        description="Exact vertex-retention probability in the closed interval [0, 1]."
    )


class HypergraphVertexContainmentResult(StrictModel):
    """The complete vertex-containment profile of a hypergraph."""

    hypergraph: FiniteHypergraph
    retention_probability: CanonicalRational
    cardinality_axis: tuple[int, ...]
    containing_subset_counts: tuple[CanonicalInteger, ...]
    total_state_count: CanonicalInteger
    success_count: CanonicalInteger
    probability: CanonicalRational


__all__ = [
    "MAX_CONTAINMENT_WORK",
    "MAX_SUBSET_STATES",
    "HypergraphVertexContainmentRequest",
    "HypergraphVertexContainmentResult",
]
