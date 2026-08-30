"""Signed induced-subgraph weight extrema kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.optimization.signed_induced_weight._models import (
    SignedInducedWeightResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_signed_induced_weight_extrema"]


def compute_signed_induced_weight_extrema(
    graph: SimpleUndirectedGraph,
    edge_weights: tuple[tuple[str, str, CanonicalRational], ...],
) -> SignedInducedWeightResult:
    """Return the min and max induced-edge total over all vertex subsets."""
    vertices = graph.vertices
    n = len(vertices)

    supplied_edges = tuple(
        (min(left, right), max(left, right)) for left, right, _ in edge_weights
    )
    if len(supplied_edges) != len(set(supplied_edges)) or set(supplied_edges) != set(
        graph.edges
    ):
        raise OperationDomainValidationError(
            location=("edge_weights",),
            code="signed_induced_weight.edge_axis",
            message="edge_weights must align one-for-one with the graph edge axis",
        )

    # Build weight lookup
    weight_map: dict[tuple[str, str], Fraction] = {}
    for u, v, w in edge_weights:
        key = (min(u, v), max(u, v))
        weight_map[key] = w.as_fraction()

    weight_values = tuple(weight_map.values())
    if weight_values and min(weight_values) < 0 < max(weight_values) and n > 19:
        raise OperationDomainValidationError(
            location=("graph",),
            code="signed_induced_weight.search_exceeded",
            message="mixed-sign induced-weight search supports at most 19 vertices",
        )

    total_edge_weight = sum(weight_values, Fraction())
    if not weight_values or min(weight_values) >= 0:
        return SignedInducedWeightResult(
            graph=graph,
            edge_weights=edge_weights,
            minimum_weight=CanonicalRational.from_fraction(Fraction()),
            minimum_witness=(),
            maximum_weight=CanonicalRational.from_fraction(total_edge_weight),
            maximum_witness=vertices if weight_values else (),
        )
    if max(weight_values) <= 0:
        return SignedInducedWeightResult(
            graph=graph,
            edge_weights=edge_weights,
            minimum_weight=CanonicalRational.from_fraction(total_edge_weight),
            minimum_witness=vertices,
            maximum_weight=CanonicalRational.from_fraction(Fraction()),
            maximum_witness=(),
        )

    def subset_weight(subset: tuple[int, ...]) -> Fraction:
        total = Fraction(0)
        for idx in range(len(subset)):
            for jdx in range(idx + 1, len(subset)):
                u = vertices[subset[idx]]
                v = vertices[subset[jdx]]
                key = (min(u, v), max(u, v))
                if key in weight_map:
                    total += weight_map[key]
        return total

    best_min = Fraction(0)
    best_min_witness: tuple[str, ...] = ()
    best_max = Fraction(0)
    best_max_witness: tuple[str, ...] = ()

    indices = list(range(n))
    for size in range(n + 1):
        for subset in combinations(indices, size):
            weight = subset_weight(subset)
            if weight < best_min:
                best_min = weight
                best_min_witness = tuple(vertices[i] for i in subset)
            if weight > best_max:
                best_max = weight
                best_max_witness = tuple(vertices[i] for i in subset)

    return SignedInducedWeightResult(
        graph=graph,
        edge_weights=edge_weights,
        minimum_weight=CanonicalRational.from_fraction(best_min),
        minimum_witness=best_min_witness,
        maximum_weight=CanonicalRational.from_fraction(best_max),
        maximum_witness=best_max_witness,
    )
