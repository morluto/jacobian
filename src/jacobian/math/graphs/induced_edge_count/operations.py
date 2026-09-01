"""Induced-edge-count profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.induced_edge_count._models import (
    InducedEdgeCountProfileResult,
    InducedEdgeCountRow,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_SUBSET_ENUMERATION = 1_000_000
MAX_RETAINED_LABEL_CHARACTERS = 1_000_000


@dataclass(frozen=True, slots=True)
class InducedEdgeCountAdmission:
    """Derived facts shared by native and catalog execution."""

    subset_count: int
    edge_count: int
    result_rows: int


def _admit_induced_edge_count_profile(
    graph: SimpleUndirectedGraph, cardinality: int
) -> InducedEdgeCountAdmission:
    """Admit one exact profile before combinations or edge scans begin."""

    if not isinstance(graph, SimpleUndirectedGraph):
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.induced_edge_count.invalid_graph",
            message="graph must be a SimpleUndirectedGraph",
        )
    if type(cardinality) is not int or cardinality < 0:
        raise OperationDomainValidationError(
            location=("cardinality",),
            code="graph.induced_edge_count.invalid_cardinality",
            message="cardinality must be a nonnegative integer",
        )
    vertex_count = len(graph.vertices)
    edge_count = len(graph.edges)
    if cardinality > vertex_count:
        raise OperationDomainValidationError(
            location=("cardinality",),
            code="graph.induced_edge_count.cardinality_too_large",
            message="cardinality must not exceed the number of graph vertices",
        )
    subset_count = comb(vertex_count, cardinality)
    work = subset_count * max(edge_count, 1)
    if work > MAX_SUBSET_ENUMERATION:
        raise OperationDomainValidationError(
            location=("cardinality",),
            code="graph.induced_edge_count.enumeration_work_exceeded",
            message=(
                "induced-edge profile exceeds the "
                f"{MAX_SUBSET_ENUMERATION}-iteration subset/edge work bound"
            ),
        )

    max_rows = min(comb(cardinality, 2) + 1, edge_count + 1, subset_count)
    source_label_characters = sum(map(len, graph.vertices)) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    largest_witness_characters = sum(
        sorted((len(vertex) for vertex in graph.vertices), reverse=True)[:cardinality]
    )
    if (
        source_label_characters + max_rows * largest_witness_characters
        > MAX_RETAINED_LABEL_CHARACTERS
    ):
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.induced_edge_count.retained_labels_exceeded",
            message="induced-edge profile exceeds the retained label-character bound",
        )
    return InducedEdgeCountAdmission(
        subset_count=subset_count,
        edge_count=edge_count,
        result_rows=max_rows,
    )


__all__ = ["MAX_RETAINED_LABEL_CHARACTERS", "compute_induced_edge_count_profile"]


def compute_induced_edge_count_profile(
    graph: SimpleUndirectedGraph,
    cardinality: int,
) -> InducedEdgeCountProfileResult:
    """Return the distribution of induced-edge counts over all k-subsets.

    For each k-element vertex subset, count the edges with both endpoints
    in the subset. The result is a histogram: for each attained count,
    the number of k-subsets having that count, and one canonical witness.
    """
    _admit_induced_edge_count_profile(graph, cardinality)
    # Establish canonical witness order once; sorting each subset would repeat
    # label comparisons for every combination.
    vertices = sorted(graph.vertices)
    edges = list(graph.edges)

    # Keep one witness and a multiplicity per attained edge count rather than
    # retaining every subset (large edgeless graphs can have nearly a million
    # subsets but only one histogram row).
    count_to_stats: dict[int, tuple[int, tuple[str, ...]]] = {}

    for subset in combinations(vertices, cardinality):
        subset_set = set(subset)
        edge_count = 0
        for a, b in edges:
            if a in subset_set and b in subset_set:
                edge_count += 1
        witness = tuple(subset)
        previous = count_to_stats.get(edge_count)
        if previous is None:
            count_to_stats[edge_count] = (1, witness)
        else:
            count, prior_witness = previous
            count_to_stats[edge_count] = (count + 1, min(prior_witness, witness))

    rows: list[InducedEdgeCountRow] = []
    for edge_count in sorted(count_to_stats):
        subset_count, witness = count_to_stats[edge_count]
        rows.append(
            InducedEdgeCountRow(
                edge_count=edge_count,
                subset_count=subset_count,
                witness=witness,
            )
        )

    return InducedEdgeCountProfileResult(
        graph=graph,
        cardinality=cardinality,
        rows=tuple(rows),
    )
