"""Induced-edge-count profile kernel."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.graphs.induced_edge_count._models import (
    InducedEdgeCountProfileResult,
    InducedEdgeCountRow,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

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
    vertices = list(graph.vertices)
    edges = list(graph.edges)

    count_to_subsets: dict[int, list[tuple[str, ...]]] = {}

    for subset in combinations(vertices, cardinality):
        subset_set = set(subset)
        edge_count = 0
        for a, b in edges:
            if a in subset_set and b in subset_set:
                edge_count += 1
        if edge_count not in count_to_subsets:
            count_to_subsets[edge_count] = []
        count_to_subsets[edge_count].append(tuple(sorted(subset)))

    rows: list[InducedEdgeCountRow] = []
    for edge_count in sorted(count_to_subsets):
        subsets = count_to_subsets[edge_count]
        witness = min(subsets)
        rows.append(
            InducedEdgeCountRow(
                edge_count=edge_count,
                subset_count=len(subsets),
                witness=witness,
            )
        )

    return InducedEdgeCountProfileResult(
        graph=graph,
        cardinality=cardinality,
        rows=tuple(rows),
    )
