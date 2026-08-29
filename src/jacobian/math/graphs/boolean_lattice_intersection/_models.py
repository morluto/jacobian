"""Typed contracts for the Boolean-lattice intersection graph operation."""

from typing import Literal

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class BooleanLatticeIntersectionRequest(StrictModel):
    """Request to construct a Boolean-lattice intersection graph."""

    n: int
    intersection_cardinality: int
    relation: Literal["INTERSECTION_EQ_THRESHOLD", "INTERSECTION_LT_THRESHOLD", "INTERSECTION_GT_THRESHOLD"]


class BooleanLatticeIntersectionResult(StrictModel):
    """The Boolean-lattice intersection graph."""

    n: int
    intersection_cardinality: int
    relation: str
    graph: SimpleUndirectedGraph


__all__ = [
    "BooleanLatticeIntersectionRequest",
    "BooleanLatticeIntersectionResult",
]
