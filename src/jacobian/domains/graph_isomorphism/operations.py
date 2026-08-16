"""Domain adapter for bounded graph isomorphism decision operations."""

from __future__ import annotations

import networkx as nx

from jacobian.contracts.graph_isomorphism import (
    GraphIsomorphismRequest,
    GraphIsomorphismResult,
)


def _build_graph(vertex_count: int, edges: tuple[tuple[int, int], ...]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(vertex_count))
    graph.add_edges_from(edges)
    return graph


def compute_isomorphism_decision(
    request: GraphIsomorphismRequest,
) -> GraphIsomorphismResult:
    """Decide graph isomorphism and return an explicit mapping certificate.

    Uses NetworkX's :func:`networkx.is_isomorphic` for the decision and
    :class:`networkx.algorithms.isomorphism.GraphMatcher` to extract one
    explicit vertex bijection when the graphs are isomorphic.  The caller
    can independently verify the returned mapping.
    """
    graph_a = _build_graph(request.graph_a.vertex_count, request.graph_a.edges)
    graph_b = _build_graph(request.graph_b.vertex_count, request.graph_b.edges)

    if not nx.is_isomorphic(graph_a, graph_b):
        return GraphIsomorphismResult(decision="NOT_ISOMORPHIC")

    matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b)
    # GraphMatcher is lazily evaluated; is_isomorphic runs the search.
    # When it returns True, mapping() yields at least one mapping.
    matcher.is_isomorphic()
    mapping = matcher.mapping
    mapping_tuple = tuple(
        sorted((source, target) for source, target in mapping.items())
    )
    return GraphIsomorphismResult(decision="ISOMORPHIC", mapping=mapping_tuple)
