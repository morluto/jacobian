"""Common-neighbour profile kernel."""

from __future__ import annotations

from jacobian.math.graphs.common_neighbor_profile._models import (
    CommonNeighborProfileResult,
    PairEntry,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_common_neighbor_profile"]


def compute_common_neighbor_profile(
    graph: SimpleUndirectedGraph,
) -> CommonNeighborProfileResult:
    """Return the common-neighbour profile of a finite simple graph.

    For every unordered pair {u, v} of distinct vertices, compute the
    canonical sorted tuple of common neighbours N(u) ∩ N(v) and its
    cardinality (the codegree). Also return the maximum codegree, the
    complete cardinality histogram, and whether the graph is C4-free
    (every pair has codegree at most 1).
    """
    vertices = graph.vertices
    n = len(vertices)

    vertex_index = {v: i for i, v in enumerate(vertices)}

    adjacency: list[set[str]] = [set() for _ in range(n)]
    for left, right in graph.edges:
        adjacency[vertex_index[left]].add(right)
        adjacency[vertex_index[right]].add(left)

    entries: list[PairEntry] = []
    histogram_dict: dict[int, int] = {}

    for i in range(n):
        for j in range(i + 1, n):
            u_label = vertices[i]
            v_label = vertices[j]
            common = sorted(adjacency[i] & adjacency[j])
            codegree = len(common)
            entries.append(
                PairEntry(
                    u=u_label,
                    v=v_label,
                    common_neighbors=tuple(common),
                    codegree=codegree,
                )
            )
            histogram_dict[codegree] = histogram_dict.get(codegree, 0) + 1

    if entries:
        max_codegree = max(entry.codegree for entry in entries)
        is_c4_free = all(entry.codegree <= 1 for entry in entries)
    else:
        max_codegree = 0
        is_c4_free = True

    if histogram_dict:
        max_key = max(histogram_dict)
        histogram = tuple(
            histogram_dict.get(k, 0) for k in range(max_key + 1)
        )
    else:
        histogram = ()

    return CommonNeighborProfileResult(
        graph=graph,
        pairs=tuple(entries),
        max_codegree=max_codegree,
        histogram=histogram,
        is_c4_free=is_c4_free,
    )
