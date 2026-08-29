"""Typed contracts for the maximal-clique hypergraph operation."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


class MaximalCliqueHypergraphRequest(StrictModel):
    """Request to construct the maximal-clique hypergraph of a graph."""

    graph: SimpleUndirectedGraph


class MaximalCliqueHypergraphResult(StrictModel):
    """The maximal-clique hypergraph of the source graph."""

    graph: SimpleUndirectedGraph
    hypergraph: FiniteHypergraph
    clique_count: int


__all__ = [
    "MaximalCliqueHypergraphRequest",
    "MaximalCliqueHypergraphResult",
]
