"""Typed contracts for uniform-subset intersection graph construction."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_GROUND_SET_SIZE = 12
MAX_VERTEX_COUNT = 256  # C(12,6) = 924 > 256, but the graph vertex cap limits this
MAX_EDGE_COUNT = 256 * 255 // 2

IntersectionRelation = Literal["INTERSECTION_LT_THRESHOLD", "INTERSECTION_EQ_THRESHOLD"]


class UniformSubsetIntersectionRequest(StrictModel):
    """Construct a graph from k-subsets of [n] with a threshold relation."""

    ground_set_size: int = Field(ge=0, le=MAX_GROUND_SET_SIZE)
    subset_cardinality: int = Field(ge=0, le=MAX_GROUND_SET_SIZE)
    threshold: int = Field(ge=0)
    relation: IntersectionRelation

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.subset_cardinality > self.ground_set_size:
            raise PydanticCustomError(
                "graph.subset_cardinality_exceeds_ground_set",
                "subset_cardinality must not exceed ground_set_size",
            )
        if self.threshold > self.subset_cardinality:
            raise PydanticCustomError(
                "graph.threshold_exceeds_subset_cardinality",
                "threshold must not exceed subset_cardinality",
            )
        return self


class UniformSubsetIntersectionResult(StrictModel):
    """The constructed uniform-subset intersection graph."""

    ground_set_size: int
    subset_cardinality: int
    threshold: int
    relation: IntersectionRelation
    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_GROUND_SET_SIZE",
    "IntersectionRelation",
    "UniformSubsetIntersectionRequest",
    "UniformSubsetIntersectionResult",
]
