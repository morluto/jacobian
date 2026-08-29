"""Typed contracts for the word-cube combinatorial-line hypergraph."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


class WordCubeRequest(StrictModel):
    """Request to construct the combinatorial-line hypergraph of [q]^d."""

    alphabet_size: int
    dimension: int


class WordCubeResult(StrictModel):
    """The combinatorial-line hypergraph of [q]^d."""

    alphabet_size: int
    dimension: int
    hypergraph: FiniteHypergraph


__all__ = ["WordCubeRequest", "WordCubeResult"]
