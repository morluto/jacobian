"""Typed contracts for the binary-union relation hypergraph."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


class BinaryUnionHypergraphRequest(StrictModel):
    """Request to construct the binary-union relation hypergraph."""

    sets: tuple[tuple[int, ...], ...]


class BinaryUnionHypergraphResult(StrictModel):
    """The 3-uniform binary-union relation hypergraph."""

    sets: tuple[tuple[int, ...], ...]
    hypergraph: FiniteHypergraph
    relation_count: int


__all__ = [
    "BinaryUnionHypergraphRequest",
    "BinaryUnionHypergraphResult",
]
