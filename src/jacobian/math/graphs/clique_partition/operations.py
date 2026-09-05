"""Exact supplied edge-clique partition checking."""

from __future__ import annotations

from jacobian.math.graphs.clique_partition._models import (
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

    adjacency: dict[str, frozenset[str]] = {
        vertex: frozenset() for vertex in graph.vertices
    }
    neighbors: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    adjacency = {vertex: frozenset(peers) for vertex, peers in neighbors.items()}

    for index, part in enumerate(parts):
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
