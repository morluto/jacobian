"""Admission planning for signed induced-weight extrema."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.optimization._models import RationalWeightedGraph

MAX_SIGNED_WEIGHT_VERTICES = 20
MAX_SIGNED_WEIGHT_EDGES = (
    MAX_SIGNED_WEIGHT_VERTICES * (MAX_SIGNED_WEIGHT_VERTICES - 1) // 2
)
MAX_SUBSET_ENUMERATION = 1 << MAX_SIGNED_WEIGHT_VERTICES
MAX_SIGNED_WEIGHT_WORK_UNITS = 25_000_000


@dataclass(frozen=True, slots=True)
class SignedInducedWeightAdmission:
    """One exact integer-scaled exhaustive-search plan."""

    denominator: int
    adjacency: tuple[tuple[tuple[int, int], ...], ...]
    candidate_subsets: int
    edge_updates: int
    integer_limbs: int
    work_units: int


def _decimal_digit_upper_bound(value: int) -> int:
    """Return a cheap conservative decimal-width bound for an integer."""

    if value == 0:
        return 1
    return (abs(value).bit_length() * 30_103) // 100_000 + 1


def admit_signed_induced_weight(
    graph: RationalWeightedGraph,
) -> SignedInducedWeightAdmission:
    """Derive one bounded exact search plan shared by native and wire calls."""

    if not isinstance(graph, RationalWeightedGraph):
        raise TypeError("signed_induced_weight_extrema expects a RationalWeightedGraph")
    order = len(graph.vertices)
    if order > MAX_SIGNED_WEIGHT_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.signed_induced_weight.vertex_bound",
            message=(
                "signed induced-weight extrema support at most "
                f"{MAX_SIGNED_WEIGHT_VERTICES} vertices"
            ),
        )
    if len(graph.edges) > MAX_SIGNED_WEIGHT_EDGES:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.signed_induced_weight.edge_bound",
            message=(
                "signed induced-weight extrema support at most "
                f"{MAX_SIGNED_WEIGHT_EDGES} edges"
            ),
        )

    fractions = tuple(edge.weight.as_fraction() for edge in graph.edges)
    denominator = lcm(*(value.denominator for value in fractions)) if fractions else 1
    if _decimal_digit_upper_bound(denominator) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.signed_induced_weight.rational_height_bound",
            message=(
                "a common denominator for the induced-weight sums may exceed the "
                f"canonical {MAX_CANONICAL_RATIONAL_DIGITS:,}-digit rational bound"
            ),
        )

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    adjacency_lists: list[list[tuple[int, int]]] = [[] for _ in graph.vertices]
    scaled_absolute_sum = 0
    for edge, value in zip(graph.edges, fractions, strict=True):
        scaled = value.numerator * (denominator // value.denominator)
        left = vertex_index[edge.endpoints[0]]
        right = vertex_index[edge.endpoints[1]]
        adjacency_lists[left].append((right, scaled))
        adjacency_lists[right].append((left, scaled))
        scaled_absolute_sum += abs(scaled)
    if _decimal_digit_upper_bound(scaled_absolute_sum) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.signed_induced_weight.rational_height_bound",
            message=(
                "an induced-weight numerator may exceed the canonical "
                f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit rational bound"
            ),
        )

    candidate_subsets = 1 << order
    degrees = tuple(len(neighbors) for neighbors in adjacency_lists)
    edge_updates = sum(
        degree * (1 << (order - vertex - 1)) for vertex, degree in enumerate(degrees)
    )
    maximum_bits = max(denominator.bit_length(), scaled_absolute_sum.bit_length(), 1)
    integer_limbs = (maximum_bits + 63) // 64
    work_units = candidate_subsets + edge_updates * integer_limbs
    if work_units > MAX_SIGNED_WEIGHT_WORK_UNITS:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.signed_induced_weight.work_budget",
            message=(
                "the exhaustive signed induced-weight search requires "
                f"{work_units:,} candidate and integer-limb update work units, "
                f"exceeding the {MAX_SIGNED_WEIGHT_WORK_UNITS:,}-unit bound"
            ),
        )

    return SignedInducedWeightAdmission(
        denominator=denominator,
        adjacency=tuple(tuple(sorted(neighbors)) for neighbors in adjacency_lists),
        candidate_subsets=candidate_subsets,
        edge_updates=edge_updates,
        integer_limbs=integer_limbs,
        work_units=work_units,
    )


__all__ = [
    "MAX_SIGNED_WEIGHT_EDGES",
    "MAX_SIGNED_WEIGHT_VERTICES",
    "MAX_SIGNED_WEIGHT_WORK_UNITS",
    "MAX_SUBSET_ENUMERATION",
    "SignedInducedWeightAdmission",
    "admit_signed_induced_weight",
]
