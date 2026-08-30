"""Induced-edge-count profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from pydantic_core import PydanticCustomError

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.induced_edge_count._models import (
    InducedEdgeCountProfileResult,
    InducedEdgeCountRow,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_SUBSET_ENUMERATION = 1_000_000


@dataclass(frozen=True, slots=True)
class InducedEdgeCountAdmission:
    """Derived facts shared by native and catalog execution."""

    subset_count: int
    edge_count: int
    result_rows: int
    result_bytes: int


def _array_size(value_sizes: tuple[int, ...]) -> int:
    return 2 + max(len(value_sizes) - 1, 0) + sum(value_sizes)


def _int_size(value: int) -> int:
    return len(encode_strict_json(value))


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

    try:
        graph_bytes = len(encode_strict_json(graph.model_dump(mode="json")))
        label_sizes = tuple(len(encode_strict_json(label)) for label in graph.vertices)
        witness_bytes = _array_size(
            tuple(sorted(label_sizes, reverse=True)[:cardinality])
        )
        max_rows = min(comb(cardinality, 2) + 1, edge_count + 1, subset_count)
        row_bytes = strict_json_object_size(
            (
                ("edge_count", _int_size(edge_count)),
                ("subset_count", _int_size(subset_count)),
                ("witness", witness_bytes),
            )
        )
        rows_bytes = _array_size((row_bytes,) * max_rows)
        result_bytes = strict_json_object_size(
            (
                ("graph", graph_bytes),
                ("cardinality", _int_size(cardinality)),
                ("rows", rows_bytes),
            )
        )
    except (ValueError, TypeError, PydanticCustomError) as error:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.induced_edge_count.source_not_canonical",
            message="graph cannot be represented in canonical JSON",
        ) from error
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.induced_edge_count.result_bytes_exceeded",
            message="induced-edge profile exceeds the canonical output-byte limit",
        )
    return InducedEdgeCountAdmission(
        subset_count=subset_count,
        edge_count=edge_count,
        result_rows=max_rows,
        result_bytes=result_bytes,
    )


__all__ = ["compute_induced_edge_count_profile"]


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
