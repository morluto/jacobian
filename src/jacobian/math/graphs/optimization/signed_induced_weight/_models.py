"""Typed contracts for the signed induced-subgraph weight operation."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class SignedInducedWeightRequest(StrictModel):
    """Request the signed induced-subgraph weight extrema."""

    graph: SimpleUndirectedGraph
    edge_weights: tuple[tuple[str, str, CanonicalRational], ...]

    @model_validator(mode="after")
    def require_exact_edge_axis(self) -> Self:
        supplied = tuple(
            (min(left, right), max(left, right)) for left, right, _ in self.edge_weights
        )
        if len(supplied) != len(set(supplied)) or set(supplied) != set(
            self.graph.edges
        ):
            raise PydanticCustomError(
                "signed_induced_weight.edge_axis",
                "edge_weights must align one-for-one with the graph edge axis",
            )
        values = tuple(weight.as_fraction() for _, _, weight in self.edge_weights)
        if values and min(values) < 0 < max(values) and len(self.graph.vertices) > 19:
            raise PydanticCustomError(
                "signed_induced_weight.search_exceeded",
                "mixed-sign induced-weight search supports at most 19 vertices",
            )
        return self


class SignedInducedWeightResult(StrictModel):
    """The exact signed induced-subgraph weight extrema."""

    graph: SimpleUndirectedGraph
    edge_weights: tuple[tuple[str, str, CanonicalRational], ...]
    minimum_weight: CanonicalRational
    minimum_witness: tuple[str, ...]
    maximum_weight: CanonicalRational
    maximum_witness: tuple[str, ...]


__all__ = [
    "SignedInducedWeightRequest",
    "SignedInducedWeightResult",
]
