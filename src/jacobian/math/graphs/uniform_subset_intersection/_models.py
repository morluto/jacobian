"""Typed contracts for the uniform-subset intersection graph operation."""

from math import comb
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class UniformSubsetIntersectionRequest(StrictModel):
    """Request to construct a uniform-subset intersection graph."""

    n: int = Field(ge=0)
    k: int = Field(ge=0)
    threshold: int
    relation: Literal[
        "INTERSECTION_LT_THRESHOLD",
        "INTERSECTION_EQ_THRESHOLD",
        "INTERSECTION_GT_THRESHOLD",
    ]

    @model_validator(mode="after")
    def require_bounded_uniform_family(self) -> Self:
        if self.k > self.n:
            raise PydanticCustomError(
                "uniform_subset.k_out_of_range", "k must satisfy 0 <= k <= n"
            )
        if comb(self.n, self.k) > 256:
            raise PydanticCustomError(
                "uniform_subset.vertex_count_exceeded",
                "uniform-subset family exceeds the 256-vertex graph carrier",
            )
        return self


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
