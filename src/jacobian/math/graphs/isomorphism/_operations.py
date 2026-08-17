"""Domain-owned graph isomorphism decision operation."""

from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.algorithms import isomorphism as nx_isomorphism

from jacobian.math.graphs.isomorphism._models import (
    GraphIsomorphismRequest,
    GraphIsomorphismResult,
    SimpleGraph,
    VertexMappingPair,
)


def _build_graph(graph: SimpleGraph) -> nx.Graph[int] | nx.DiGraph[int]:
    """Build a NetworkX graph from a wire ``SimpleGraph``."""
    g: nx.Graph[int] | nx.DiGraph[int] = nx.DiGraph() if graph.directed else nx.Graph()
    g.add_nodes_from(range(graph.vertex_count))
    for source, target in graph.edges:
        g.add_edge(source, target)
    return g


def _vertex_mapping(
    graph_a: SimpleGraph,
    graph_b: SimpleGraph,
) -> list[VertexMappingPair] | None:
    """Return an explicit isomorphism witness, or ``None`` when absent.

    Uses NetworkX's ``GraphMatcher``/``DiGraphMatcher`` so the returned
    mapping is a concrete bijection the caller can independently verify.
    """
    g_a = _build_graph(graph_a)
    g_b = _build_graph(graph_b)
    if graph_a.directed:
        matcher: Any = nx_isomorphism.DiGraphMatcher(g_a, g_b)
    else:
        matcher = nx_isomorphism.GraphMatcher(g_a, g_b)
    if not matcher.is_isomorphic():
        return None
    mapping = next(matcher.isomorphisms_iter())
    return [
        VertexMappingPair(from_vertex=src, to_vertex=dst)
        for src, dst in sorted(mapping.items())
    ]


def decide_graph_isomorphism(
    request: GraphIsomorphismRequest,
) -> GraphIsomorphismResult:
    """Decide whether two simple graphs are isomorphic."""
    # Fail closed on mismatched vertex counts (also enforced at the contract
    # level, but keep a defense-in-depth check at the domain boundary).
    if request.graph_a.vertex_count != request.graph_b.vertex_count:
        raise ValueError("both graphs must have the same vertex count")
    mapping = _vertex_mapping(request.graph_a, request.graph_b)
    if mapping is None:
        return GraphIsomorphismResult(status="NOT_ISOMORPHIC", vertex_mapping=())
    return GraphIsomorphismResult(
        status="ISOMORPHIC",
        vertex_mapping=tuple(mapping),
    )
