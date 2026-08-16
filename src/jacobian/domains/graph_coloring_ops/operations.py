"""Domain adapter for graph coloring and independent set operations."""

from __future__ import annotations

from typing import Any

import networkx as nx

from jacobian.contracts.graph_coloring_ops import (
    GraphEdgeList,
    KColorabilityRequest,
    KColorabilityResult,
    MaximumIndependentSetRequest,
    MaximumIndependentSetResult,
)


def _build_graph(graph: GraphEdgeList) -> nx.Graph[int]:
    g: nx.Graph[Any] = nx.Graph[Any]()
    g.add_nodes_from(range(graph.vertex_count))
    g.add_edges_from(graph.edges)
    return g


def compute_k_colorability(request: KColorabilityRequest) -> KColorabilityResult:
    g = _build_graph(request.graph)
    try:
        coloring_dict = nx.coloring.greedy_color(g, strategy="largest_first")
        num_colors = len(set(coloring_dict.values()))
        if num_colors <= request.colors:
            coloring = [
                coloring_dict.get(i, 0) for i in range(request.graph.vertex_count)
            ]
            return KColorabilityResult(
                colorable=True,
                coloring=tuple(coloring),
                vertex_count=request.graph.vertex_count,
                colors=request.colors,
            )
    except Exception:
        pass
    return KColorabilityResult(
        colorable=False,
        vertex_count=request.graph.vertex_count,
        colors=request.colors,
    )


def compute_maximum_independent_set(
    request: MaximumIndependentSetRequest,
) -> MaximumIndependentSetResult:
    g = _build_graph(request.graph)
    from networkx.algorithms.approximation import maximum_independent_set

    iset = maximum_independent_set(g)
    return MaximumIndependentSetResult(
        independent_set=tuple(sorted(iset)),
        cardinality=len(iset),
    )
