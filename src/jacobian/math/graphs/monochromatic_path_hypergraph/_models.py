"""Typed contracts for the monochromatic path hypergraph operation."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph


class MonochromaticPathRequest(StrictModel):
    """Request to construct monochromatic path candidate hypergraphs."""

    graph: ColoredUndirectedGraph


class MonochromaticPathResult(StrictModel):
    """One per-colour monochromatic path hypergraph."""

    color: str
    hypergraph: FiniteHypergraph


class MonochromaticPathHypergraphResult(StrictModel):
    """Complete monochromatic path candidate hypergraph family."""

    graph: ColoredUndirectedGraph
    per_color: tuple[MonochromaticPathResult, ...]


__all__ = [
    "MonochromaticPathHypergraphResult",
    "MonochromaticPathRequest",
    "MonochromaticPathResult",
]
