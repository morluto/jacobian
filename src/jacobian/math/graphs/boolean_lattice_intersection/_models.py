"""Typed contracts for the Boolean-lattice intersection graph operation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_GROUND_SET_SIZE = 8
MAX_THRESHOLD = (1 << 53) - 1

IntersectionRelation = Literal["INTERSECTION_EQ", "INTERSECTION_LT", "INTERSECTION_GT"]


def _validate_intersection_request(
    ground_set_size: int, threshold: int, relation: IntersectionRelation
) -> None:
    if not isinstance(ground_set_size, int) or isinstance(ground_set_size, bool):
        raise PydanticCustomError(
            "boolean_lattice.ground_set_size_type",
            "ground_set_size must be an integer",
        )
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise PydanticCustomError(
            "boolean_lattice.threshold_type", "threshold must be an integer"
        )
    if not 0 <= ground_set_size <= MAX_GROUND_SET_SIZE:
        raise PydanticCustomError(
            "boolean_lattice.ground_set_size_too_large",
            f"ground_set_size must be between 0 and {MAX_GROUND_SET_SIZE}",
        )
    if not isinstance(relation, str) or relation not in {
        "INTERSECTION_EQ",
        "INTERSECTION_LT",
        "INTERSECTION_GT",
    }:
        raise PydanticCustomError(
            "boolean_lattice.invalid_relation",
            "relation must be INTERSECTION_EQ, INTERSECTION_LT, or INTERSECTION_GT",
        )
    if threshold < 0:
        raise PydanticCustomError(
            "boolean_lattice.threshold_exceeds_n",
            "threshold must be nonnegative",
        )
    if threshold > MAX_THRESHOLD:
        raise PydanticCustomError(
            "boolean_lattice.threshold_exceeds_json_integer",
            f"threshold must not exceed {MAX_THRESHOLD}",
        )
    vertex_count = 1 << ground_set_size
    edge_count = vertex_count * (vertex_count - 1) // 2
    if edge_count > 32_640:
        raise PydanticCustomError(
            "boolean_lattice.edge_envelope",
            "the Boolean-lattice graph exceeds the simple-graph edge envelope",
        )


class BooleanLatticeIntersectionRequest(StrictModel):
    """Construct a graph on the Boolean lattice 2^[n] with an intersection relation."""

    ground_set_size: int = Field(ge=0, le=MAX_GROUND_SET_SIZE)
    threshold: int = Field(ge=0, le=MAX_THRESHOLD)
    relation: IntersectionRelation


class BooleanLatticeIntersectionResult(StrictModel):
    """The Boolean-lattice intersection graph."""

    ground_set_size: int
    threshold: int
    relation: IntersectionRelation
    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_GROUND_SET_SIZE",
    "MAX_THRESHOLD",
    "BooleanLatticeIntersectionRequest",
    "BooleanLatticeIntersectionResult",
    "IntersectionRelation",
    "_validate_intersection_request",
]
