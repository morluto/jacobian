"""Typed contracts for the maximal-clique hypergraph operation."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_GRAPH_VERTICES = 256


class MaximalCliqueHypergraphRequest(StrictModel):
    """Request to construct the maximal-clique hypergraph of a graph."""

    graph: SimpleUndirectedGraph


class MaximalCliqueHypergraphResult(StrictModel):
    """The maximal-clique hypergraph of a graph."""

    graph: SimpleUndirectedGraph
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_GRAPH_VERTICES",
    "MaximalCliqueHypergraphRequest",
    "MaximalCliqueHypergraphResult",
]
