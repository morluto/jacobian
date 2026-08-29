"""Exact signed induced-edge weight extrema over all vertex subsets."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.optimization._models import RationalWeightedGraph
from jacobian.math.graphs.signed_induced_weight._models import (
    SignedInducedWeightResult,
    WeightExtremum,
)

__all__ = ["signed_induced_weight_extrema"]


def signed_induced_weight_extrema(
    graph: RationalWeightedGraph,
) -> SignedInducedWeightResult:
    """Return exact min and max signed induced-edge weight over all subsets.

    For a vertex subset S, the value is
    ``sum(weight(u,v) : {u,v} is an edge and both endpoints are in S)``.
    Empty and singleton subsets have weight zero. Ties are broken by the
    lexicographically least selected vertex axis.
    """
    vertices = list(graph.vertices)
    n = len(vertices)
    edge_data = [
        (edge.endpoints[0], edge.endpoints[1], edge.weight.as_fraction())
        for edge in graph.edges
    ]

    min_val = Fraction(0)
    min_witness: tuple[str, ...] = ()
    max_val = Fraction(0)
    max_witness: tuple[str, ...] = ()

    for mask in range(1 << n):
        selected = tuple(vertices[i] for i in range(n) if mask & (1 << i))
        selected_set = set(selected)
        weight = Fraction(0)
        for a, b, w in edge_data:
            if a in selected_set and b in selected_set:
                weight += w

        if mask == 0:
            min_val = weight
            min_witness = selected
            max_val = weight
            max_witness = selected
        else:
            if weight < min_val or (weight == min_val and selected < min_witness):
                min_val = weight
                min_witness = selected
            if weight > max_val or (weight == max_val and selected < max_witness):
                max_val = weight
                max_witness = selected

    return SignedInducedWeightResult(
        graph=graph,
        minimum=WeightExtremum(
            value=CanonicalRational.from_fraction(min_val),
            witness_vertices=min_witness,
        ),
        maximum=WeightExtremum(
            value=CanonicalRational.from_fraction(max_val),
            witness_vertices=max_witness,
        ),
    )
