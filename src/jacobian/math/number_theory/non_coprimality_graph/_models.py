"""Typed contracts for the non-coprimality graph operation."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet
from jacobian.math.graphs.values import MAX_GRAPH_LABEL_BYTES, SimpleUndirectedGraph

MAX_INTEGERS = 256
MAX_INTEGER_DIGITS = MAX_GRAPH_LABEL_BYTES


class NonCoprimalityGraphRequest(StrictModel):
    """Request to construct the non-coprimality graph of a set of integers."""

    integers: FiniteIntegerSet


class NonCoprimalityGraphResult(StrictModel):
    """The non-coprimality graph of a set of integers."""

    integers: FiniteIntegerSet
    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_INTEGERS",
    "MAX_INTEGER_DIGITS",
    "NonCoprimalityGraphRequest",
    "NonCoprimalityGraphResult",
]
