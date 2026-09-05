"""Admission planning for signed clique-weight maximization."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm

import networkx as nx

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs._networkx import biconnected_components
from jacobian.math.graphs.optimization._models import RationalWeightedGraph

# A clique lies inside one biconnected block, so the Gray-code search runs
# per block: 2**m subsets tracking the induced weight and the nonedge
# count incrementally. One 20-vertex component is the exhaustive envelope.
MAX_SIGNED_CLIQUE_VERTICES = 32
MAX_SIGNED_CLIQUE_EDGES = 496
MAX_SIGNED_CLIQUE_BLOCK_VERTICES = 20
MAX_SIGNED_CLIQUE_WORK_UNITS = 25_000_000


@dataclass(frozen=True, slots=True)
class SignedCliqueComponent:
    """One biconnected vertex block with its local dual-accumulator charge."""

    vertices: tuple[int, ...]
    adjacency: tuple[tuple[tuple[int, int], ...], ...]
    candidate_subsets: int
    edge_updates: int


@dataclass(frozen=True, slots=True)
class SignedCliqueWeightAdmission:
    """One exact integer-scaled block-bounded clique-search plan."""

    denominator: int
    components: tuple[SignedCliqueComponent, ...]
    integer_limbs: int
    work_units: int


def _decimal_digit_upper_bound(value: int) -> int:
    """Return a cheap conservative decimal-width bound for an integer."""

    if value == 0:
        return 1
    return (abs(value).bit_length() * 30_103) // 100_000 + 1


def _block_groups(
    order: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    """Return blocks including bridges, in source-vertex order.

    Every clique of order at least three is biconnected; every edge lies
    in exactly one block. Isolated vertices cannot supply eligible cliques.
    """

    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(range(order))
    graph.add_edges_from(edges)
    return tuple(
        sorted(tuple(sorted(block)) for block in biconnected_components(graph))
    )


def admit_signed_clique_weight(
    graph: RationalWeightedGraph,
) -> SignedCliqueWeightAdmission:
    """Derive one bounded block-search plan shared by native and wire calls."""

    if not isinstance(graph, RationalWeightedGraph):
        raise TypeError("signed_clique_weight_maximum expects a RationalWeightedGraph")
    order = len(graph.vertices)
    if order > MAX_SIGNED_CLIQUE_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.signed_clique_weight.vertex_bound",
            message=(
                "signed clique-weight maximization supports at most "
                f"{MAX_SIGNED_CLIQUE_VERTICES} vertices"
            ),
        )
    if len(graph.edges) > MAX_SIGNED_CLIQUE_EDGES:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.signed_clique_weight.edge_bound",
            message=(
                "signed clique-weight maximization supports at most "
                f"{MAX_SIGNED_CLIQUE_EDGES} edges"
            ),
        )

    fractions = tuple(edge.weight.as_fraction() for edge in graph.edges)
    denominator = lcm(*(value.denominator for value in fractions)) if fractions else 1
    if _decimal_digit_upper_bound(denominator) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.signed_clique_weight.rational_height_bound",
            message=(
                "a common denominator for the clique-weight sums may exceed the "
                f"canonical {MAX_CANONICAL_RATIONAL_DIGITS:,}-digit rational bound"
            ),
        )

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    scaled = tuple(
        value.numerator * (denominator // value.denominator) for value in fractions
    )
    scaled_absolute_sum = sum(abs(entry) for entry in scaled)
    if _decimal_digit_upper_bound(scaled_absolute_sum) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.signed_clique_weight.rational_height_bound",
            message=(
                "a clique-weight numerator may exceed the canonical "
                f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit rational bound"
            ),
        )

    pairs = tuple(
        (vertex_index[edge.endpoints[0]], vertex_index[edge.endpoints[1]])
        for edge in graph.edges
    )
    maximum_bits = max(denominator.bit_length(), scaled_absolute_sum.bit_length(), 1)
    integer_limbs = (maximum_bits + 63) // 64
    # Decomposition is linear; constructing each local adjacency scans all edges.
    presolve_work = order + 3 * len(graph.edges)
    components: list[SignedCliqueComponent] = []
    component_work = 0
    for group in _block_groups(order, pairs):
        if len(group) == 1:
            continue
        if len(group) > MAX_SIGNED_CLIQUE_BLOCK_VERTICES:
            raise OperationDomainValidationError(
                location=("graph",),
                code="graph.signed_clique_weight.work_budget",
                message=(
                    "a biconnected block exceeds the "
                    f"{MAX_SIGNED_CLIQUE_BLOCK_VERTICES}-vertex exhaustive "
                    "search envelope"
                ),
            )
        position = {global_index: local for local, global_index in enumerate(group)}
        adjacency: list[list[tuple[int, int]]] = [[] for _ in group]
        for (left, right), weight in zip(pairs, scaled, strict=True):
            if left in position and right in position:
                adjacency[position[left]].append((position[right], weight))
                adjacency[position[right]].append((position[left], weight))
        size = len(group)
        # Each Gray-code step updates the weight sum and the nonedge count
        # over at most size - 1 accumulators each.
        edge_updates = 2 * sum(
            (size - 1) * (1 << (size - local - 1)) for local in range(size)
        )
        work = (1 << size) + edge_updates * integer_limbs
        component_work += work + len(pairs)
        components.append(
            SignedCliqueComponent(
                vertices=group,
                adjacency=tuple(tuple(sorted(peers)) for peers in adjacency),
                candidate_subsets=1 << size,
                edge_updates=edge_updates,
            )
        )
    work_units = presolve_work + component_work
    if work_units > MAX_SIGNED_CLIQUE_WORK_UNITS:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.signed_clique_weight.work_budget",
            message=(
                "the block-bounded clique-weight search requires "
                f"{work_units:,} candidate and integer-limb update work units, "
                f"exceeding the {MAX_SIGNED_CLIQUE_WORK_UNITS:,}-unit bound"
            ),
        )
    return SignedCliqueWeightAdmission(
        denominator=denominator,
        components=tuple(components),
        integer_limbs=integer_limbs,
        work_units=work_units,
    )


__all__ = [
    "MAX_SIGNED_CLIQUE_BLOCK_VERTICES",
    "MAX_SIGNED_CLIQUE_EDGES",
    "MAX_SIGNED_CLIQUE_VERTICES",
    "MAX_SIGNED_CLIQUE_WORK_UNITS",
    "SignedCliqueComponent",
    "SignedCliqueWeightAdmission",
    "admit_signed_clique_weight",
]
