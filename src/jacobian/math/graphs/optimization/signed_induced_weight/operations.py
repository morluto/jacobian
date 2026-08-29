"""Signed induced-subgraph weight extrema kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
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

    # Build weight lookup
    weight_map: dict[tuple[str, str], Fraction] = {}
    for u, v, w in edge_weights:
        key = (min(u, v), max(u, v))
        weight_map[key] = w.as_fraction()

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
        minimum_weight=CanonicalRational.from_fraction(best_min),
        minimum_witness=best_min_witness,
        maximum_weight=CanonicalRational.from_fraction(best_max),
        maximum_witness=best_max_witness,
    )
