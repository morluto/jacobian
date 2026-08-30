"""Typed contracts for the monochromatic clique hypergraph operation."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    ColoredUndirectedGraph,
)

MAX_VERTICES = MAX_INDEXED_SIMPLE_GRAPH_VERTICES
MAX_CLIQUE_ORDER = MAX_VERTICES


class MonochromaticCliqueHypergraphRequest(StrictModel):
    """Request to construct the monochromatic clique hypergraph."""

    colored_graph: ColoredUndirectedGraph
    clique_order: StrictInt = Field(ge=2, le=MAX_CLIQUE_ORDER)


class MonochromaticCliqueHypergraphResult(StrictModel):
    """The monochromatic clique hypergraph of a coloured complete graph."""

    colored_graph: ColoredUndirectedGraph
    clique_order: StrictInt = Field(ge=2, le=MAX_CLIQUE_ORDER)
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_CLIQUE_ORDER",
    "MAX_VERTICES",
    "MonochromaticCliqueHypergraphRequest",
    "MonochromaticCliqueHypergraphResult",
]
