"""Common-neighbour profile kernel."""

from __future__ import annotations

from jacobian.math.graphs.common_neighbor_profile._models import (
    CommonNeighborProfileResult,
    CommonNeighborRow,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_common_neighbor_profile"]


def compute_common_neighbor_profile(
    graph: SimpleUndirectedGraph,
) -> CommonNeighborProfileResult:
    """Return the complete common-neighbour profile of a simple graph.

    For every unordered pair of distinct vertices, return the sorted
    set of common neighbours, its cardinality (codegree), in canonical
    source-vertex order.
    """
    vertices = list(graph.vertices)
    adjacency: dict[str, set[str]] = {v: set() for v in vertices}
    for a, b in graph.edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    rows: list[CommonNeighborRow] = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            u, v = vertices[i], vertices[j]
            common = sorted(adjacency[u] & adjacency[v])
            rows.append(
                CommonNeighborRow(
                    vertex_u=u,
                    vertex_v=v,
                    common_neighbors=tuple(common),
                    codegree=len(common),
                )
            )
    return CommonNeighborProfileResult(graph=graph, rows=tuple(rows))
