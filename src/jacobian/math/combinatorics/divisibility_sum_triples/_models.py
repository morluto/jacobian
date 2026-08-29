"""Typed contracts for the divisibility-sum triples hypergraph operation."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


class DivisibilitySumTriplesRequest(StrictModel):
    """Request to construct the divisibility-sum triples hypergraph."""

    lower: int
    upper: int


class DivisibilitySumTriplesResult(StrictModel):
    """The canonical divisibility-sum triples hypergraph."""

    lower: int
    upper: int
    hypergraph: FiniteHypergraph


__all__ = [
    "DivisibilitySumTriplesRequest",
    "DivisibilitySumTriplesResult",
]
