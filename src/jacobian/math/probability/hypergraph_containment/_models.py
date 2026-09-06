"""Typed contracts for the hypergraph vertex containment operation."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

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

    @model_validator(mode="after")
    def bind_cardinality_axis(self) -> Self:
        expected = tuple(range(len(self.hypergraph.vertices) + 1))
        if self.cardinality_axis != expected:
            raise PydanticCustomError(
                "hypergraph_containment.cardinality_axis_mismatch",
                "cardinality_axis must cover every subset cardinality in order",
            )
        if len(self.containing_subset_counts) != len(expected):
            raise PydanticCustomError(
                "hypergraph_containment.profile_length_mismatch",
                "containing subset counts must align with cardinality_axis",
            )
        return self


__all__ = [
    "MAX_CONTAINMENT_WORK",
    "MAX_SUBSET_STATES",
    "HypergraphVertexContainmentRequest",
    "HypergraphVertexContainmentResult",
]
