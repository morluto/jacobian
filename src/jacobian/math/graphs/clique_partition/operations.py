"""Exact supplied edge-clique partition checking."""

from __future__ import annotations

import json

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.clique_partition._models import (
    MAX_PARTITION_OUTPUT_BYTES,
    MAX_PARTITION_PAIR_WORK,
    MAX_PARTITION_VERTEX_REFERENCES,
    EdgeCliquePartitionRequest,
    EdgeCliquePartitionResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["check_edge_clique_partition"]


def check_edge_clique_partition(
    graph: SimpleUndirectedGraph,
    parts: tuple[tuple[str, ...], ...],
) -> EdgeCliquePartitionResult:
    """Check supplied vertex subsets as an edge partition into cliques.

    Every part must be a clique of order at least two and every graph edge
    must occur in exactly one part's pair set. The first failure wins, in
    part order for clique checks and lexicographic edge order for coverage.
    Part order never affects validity. Work is quadratic in part sizes with
    adjacency-set lookup.
    """

    _admit_partition(graph, parts)
    return _check_partition(graph, parts)


def _admit_partition(
    graph: SimpleUndirectedGraph, parts: tuple[tuple[str, ...], ...]
) -> None:
    # Structural validation also protects callers of the native function.
    EdgeCliquePartitionRequest(graph=graph, parts=parts)
    references = sum(map(len, parts))
    pair_work = sum(len(part) * (len(part) - 1) // 2 for part in parts)
    if (
        references > MAX_PARTITION_VERTEX_REFERENCES
        or pair_work > MAX_PARTITION_PAIR_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("parts",),
            code="graph.clique_partition.work_bound",
            message=f"partition references={references}, pair_work={pair_work}; limits {MAX_PARTITION_VERTEX_REFERENCES}, {MAX_PARTITION_PAIR_WORK}",
        )
    label_bytes = {
        v: len(json.dumps(v, ensure_ascii=False).encode("utf-8"))
        for v in graph.vertices
    }
    # Retained source and parts dominate output; reserve all covering indices
    # and fixed diagnostic keys, even though only one failure is returned.
    output_bytes = len(graph.model_dump_json().encode("utf-8")) + 2048
    output_bytes += sum(2 + sum(label_bytes[v] + 1 for v in part) for part in parts)
    output_bytes += len(parts) * (len(str(len(parts))) + 1)
    for name, work, limit in (
        ("vertex_references", references, MAX_PARTITION_VERTEX_REFERENCES),
        ("pair_work", pair_work, MAX_PARTITION_PAIR_WORK),
        ("output_bytes", output_bytes, MAX_PARTITION_OUTPUT_BYTES),
    ):
        if work > limit:
            raise OperationResourceAdmissionError(
                location=("parts",),
                code=f"graph.clique_partition.{name}_bound",
                message=f"partition {name}={work} exceeds limit={limit}",
            )
    request_checkpoint("before edge-clique partition checking")


def _check_partition(
    graph: SimpleUndirectedGraph, parts: tuple[tuple[str, ...], ...]
) -> EdgeCliquePartitionResult:
    neighbors: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    adjacency = {vertex: frozenset(peers) for vertex, peers in neighbors.items()}

    for index, part in enumerate(parts):
        request_checkpoint("during edge-clique partition checking")
        members = list(part)
        for left_position in range(len(members)):
            for right_position in range(left_position + 1, len(members)):
                left, right = members[left_position], members[right_position]
                if right not in adjacency[left]:
                    first, second = (left, right) if left < right else (right, left)
                    return EdgeCliquePartitionResult._from_kernel(
                        graph=graph,
                        parts=parts,
                        is_partition=False,
                        failing_part=index,
                        failing_nonedge=(first, second),
                    )

    coverage: dict[tuple[str, str], list[int]] = {edge: [] for edge in graph.edges}
    for index, part in enumerate(parts):
        request_checkpoint("during edge-clique partition coverage")
        members = list(part)
        for left_position in range(len(members)):
            for right_position in range(left_position + 1, len(members)):
                left, right = members[left_position], members[right_position]
                edge = (left, right) if left < right else (right, left)
                if edge in coverage:
                    coverage[edge].append(index)
    for edge in sorted(coverage):
        if not coverage[edge]:
            return EdgeCliquePartitionResult._from_kernel(
                graph=graph,
                parts=parts,
                is_partition=False,
                uncovered_edge=edge,
            )
    for edge in sorted(coverage):
        if len(coverage[edge]) > 1:
            return EdgeCliquePartitionResult._from_kernel(
                graph=graph,
                parts=parts,
                is_partition=False,
                overcovered_edge=edge,
                overcovering_parts=tuple(coverage[edge]),
            )
    return EdgeCliquePartitionResult._from_kernel(
        graph=graph, parts=parts, is_partition=True
    )
