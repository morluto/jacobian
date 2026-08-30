"""Exact signed induced-edge weight extrema over all vertex subsets."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.optimization._models import RationalWeightedGraph
from jacobian.math.graphs.signed_induced_weight._bounds import (
    admit_signed_induced_weight,
)
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
    admission = admit_signed_induced_weight(graph)
    selected = [False] * len(graph.vertices)
    current_value = 0
    minimum_value = 0
    min_witness: tuple[str, ...] = ()
    maximum_value = 0
    max_witness: tuple[str, ...] = ()
    previous_gray = 0

    for step in range(1, admission.candidate_subsets):
        gray = step ^ (step >> 1)
        changed = (gray ^ previous_gray).bit_length() - 1
        if selected[changed]:
            selected[changed] = False
            current_value -= sum(
                weight
                for neighbor, weight in admission.adjacency[changed]
                if selected[neighbor]
            )
        else:
            current_value += sum(
                weight
                for neighbor, weight in admission.adjacency[changed]
                if selected[neighbor]
            )
            selected[changed] = True
        previous_gray = gray

        if current_value <= minimum_value or current_value >= maximum_value:
            witness = tuple(
                vertex
                for vertex, is_selected in zip(graph.vertices, selected, strict=True)
                if is_selected
            )
            if current_value < minimum_value or (
                current_value == minimum_value and witness < min_witness
            ):
                minimum_value = current_value
                min_witness = witness
            if current_value > maximum_value or (
                current_value == maximum_value and witness < max_witness
            ):
                maximum_value = current_value
                max_witness = witness

    return SignedInducedWeightResult(
        graph=graph,
        minimum=WeightExtremum(
            value=CanonicalRational.from_fraction(
                Fraction(minimum_value, admission.denominator)
            ),
            witness_vertices=min_witness,
        ),
        maximum=WeightExtremum(
            value=CanonicalRational.from_fraction(
                Fraction(maximum_value, admission.denominator)
            ),
            witness_vertices=max_witness,
        ),
    )
