"""Typed contracts for the Boolean-lattice intersection graph operation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_GROUND_SET_SIZE = 6

IntersectionRelation = Literal["INTERSECTION_EQ", "INTERSECTION_LT", "INTERSECTION_GT"]


class BooleanLatticeIntersectionRequest(StrictModel):
    """Construct a graph on the Boolean lattice 2^[n] with an intersection relation."""

    ground_set_size: int = Field(ge=0, le=MAX_GROUND_SET_SIZE)
    threshold: int = Field(ge=0)
    relation: IntersectionRelation

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.threshold > self.ground_set_size:
            raise PydanticCustomError(
                "boolean_lattice.threshold_exceeds_n",
                "threshold must not exceed ground_set_size",
            )
        return self


class BooleanLatticeIntersectionResult(StrictModel):
    """The Boolean-lattice intersection graph."""

    ground_set_size: int
    threshold: int
    relation: IntersectionRelation
    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_GROUND_SET_SIZE",
    "BooleanLatticeIntersectionRequest",
    "BooleanLatticeIntersectionResult",
    "IntersectionRelation",
]
