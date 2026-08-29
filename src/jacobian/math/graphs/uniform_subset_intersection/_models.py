"""Typed contracts for the uniform-subset intersection graph operation."""

from typing import Literal

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class UniformSubsetIntersectionRequest(StrictModel):
    """Request to construct a uniform-subset intersection graph."""

    n: int
    k: int
    threshold: int
    relation: Literal["INTERSECTION_LT_THRESHOLD", "INTERSECTION_EQ_THRESHOLD", "INTERSECTION_GT_THRESHOLD"]


class UniformSubsetIntersectionResult(StrictModel):
    """The uniform-subset intersection graph."""

    n: int
    k: int
    threshold: int
    relation: str
    graph: SimpleUndirectedGraph


__all__ = [
    "UniformSubsetIntersectionRequest",
    "UniformSubsetIntersectionResult",
]
