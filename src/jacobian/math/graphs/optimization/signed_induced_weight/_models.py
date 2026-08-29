"""Typed contracts for the signed induced-subgraph weight operation."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class SignedInducedWeightRequest(StrictModel):
    """Request the signed induced-subgraph weight extrema."""

    graph: SimpleUndirectedGraph
    edge_weights: tuple[tuple[str, str, CanonicalRational], ...]


class SignedInducedWeightResult(StrictModel):
    """The exact signed induced-subgraph weight extrema."""

    graph: SimpleUndirectedGraph
    minimum_weight: CanonicalRational
    minimum_witness: tuple[str, ...]
    maximum_weight: CanonicalRational
    maximum_witness: tuple[str, ...]


__all__ = [
    "SignedInducedWeightRequest",
    "SignedInducedWeightResult",
]
