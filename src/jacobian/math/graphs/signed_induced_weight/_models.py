"""Typed contracts for signed induced-weight extrema."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.graphs.optimization._models import (
    MAX_GRAPH_WEIGHT_DIGITS,
    RationalWeightedGraph,
    require_bounded_rational,
)

MAX_SIGNED_WEIGHT_VERTICES = 20
MAX_SIGNED_WEIGHT_EDGES = 496  # C(20,2)
MAX_SUBSET_ENUMERATION = 1 << MAX_SIGNED_WEIGHT_VERTICES  # 2^20


class SignedInducedWeightRequest(StrictModel):
    """Request for exact signed induced-edge weight extrema."""

    graph: RationalWeightedGraph

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if len(self.graph.vertices) > MAX_SIGNED_WEIGHT_VERTICES:
            raise PydanticCustomError(
                "graph.signed_induced_weight_too_many_vertices",
                f"at most {MAX_SIGNED_WEIGHT_VERTICES} vertices are supported",
            )
        if len(self.graph.edges) > MAX_SIGNED_WEIGHT_EDGES:
            raise PydanticCustomError(
                "graph.signed_induced_weight_too_many_edges",
                f"at most {MAX_SIGNED_WEIGHT_EDGES} edges are supported",
            )
        for edge in self.graph.edges:
            require_bounded_rational(
                edge.weight,
                max_digits=MAX_GRAPH_WEIGHT_DIGITS,
                label="edge weight",
            )
        return self


class WeightExtremum(StrictModel):
    """One extremum (min or max) with a witness vertex subset."""

    value: CanonicalRational
    witness_vertices: tuple[str, ...]


class SignedInducedWeightResult(StrictModel):
    """Exact min and max of the signed induced-edge weight over all subsets."""

    graph: RationalWeightedGraph
    minimum: WeightExtremum
    maximum: WeightExtremum


__all__ = [
    "MAX_SIGNED_WEIGHT_EDGES",
    "MAX_SIGNED_WEIGHT_VERTICES",
    "SignedInducedWeightRequest",
    "SignedInducedWeightResult",
    "WeightExtremum",
]
