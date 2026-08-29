"""Typed contracts for the non-coprimality conflict-graph operation."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteIntegerSet,
)

MAX_NON_COPRIMALITY_GRAPH_VERTICES = 256


class NonCoprimalityGraphRequest(StrictModel):
    """A finite set of positive integers whose non-coprimality graph is sought."""

    elements: FiniteIntegerSet


class NonCoprimalityGraphResult(StrictModel):
    """The canonical simple conflict graph joining pairs with gcd > 1."""

    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_NON_COPRIMALITY_GRAPH_VERTICES",
    "NonCoprimalityGraphRequest",
    "NonCoprimalityGraphResult",
]
