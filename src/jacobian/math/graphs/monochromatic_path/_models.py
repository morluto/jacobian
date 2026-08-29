"""Typed contracts for the monochromatic path hypergraph operation."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph

MAX_VERTICES = 12


class MonochromaticPathRequest(StrictModel):
    """Request for the monochromatic path hypergraphs of a coloured graph."""

    graph: ColoredUndirectedGraph


class MonochromaticPathResult(StrictModel):
    """The monochromatic path hypergraphs of a coloured graph."""

    graph: ColoredUndirectedGraph
    colour_to_hypergraph: dict[str, FiniteHypergraph]


__all__ = [
    "MAX_VERTICES",
    "MonochromaticPathRequest",
    "MonochromaticPathResult",
]
