"""Domain-owned directed graph operations."""

from __future__ import annotations

from typing import Any

import networkx as nx

from jacobian.math.directed_graph._models import (
    AcyclicOrderRequest,
    AcyclicOrderResult,
    CondensationEdge,
    CondensationRequest,
    CondensationResult,
    DirectedGraph,
    ReachabilityRequest,
    ReachabilityResult,
    StronglyConnectedComponentsRequest,
    StronglyConnectedComponentsResult,
)


def _build_digraph(graph: DirectedGraph) -> nx.DiGraph[int]:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(graph.vertex_count))
    for source, target in graph.edges:
        g.add_edge(source, target)
    return g


def compute_reachability(request: ReachabilityRequest) -> ReachabilityResult:
    """Determine which vertices are reachable from the source vertex.

    A vertex is reachable if there is a directed path from source to that
    vertex. The source itself is always considered reachable.
    """
    g = _build_digraph(request.graph)
    descendants = nx.descendants(g, request.source)
    reachable = frozenset(descendants) | {request.source}
    unreachable = frozenset(range(request.graph.vertex_count)) - reachable
    return ReachabilityResult(
        source=request.source,
        reachable=tuple(sorted(reachable)),
        unreachable=tuple(sorted(unreachable)),
    )


def compute_strongly_connected_components(
    request: StronglyConnectedComponentsRequest,
) -> StronglyConnectedComponentsResult:
    """Partition the graph into strongly connected components.

    Components are returned in the order NetworkX yields them; each
    component's vertices are sorted for determinism.
    """
    g = _build_digraph(request.graph)
    sccs = list(nx.strongly_connected_components(g))
    components = tuple(tuple(sorted(component)) for component in sccs)
    return StronglyConnectedComponentsResult(
        component_count=len(components),
        components=components,
    )


def compute_condensation(request: CondensationRequest) -> CondensationResult:
    """Compute the condensation of the graph.

    The condensation is the DAG whose vertices are the strongly connected
    components of the original graph. Condensation vertex ``i`` corresponds to
    the ``i``-th strongly connected component returned by NetworkX (and
    reported in the ``components`` field).
    """
    g = _build_digraph(request.graph)
    sccs = list(nx.strongly_connected_components(g))
    condensation = nx.condensation(g, sccs)

    components = tuple(tuple(sorted(component)) for component in sccs)

    edges: list[CondensationEdge] = [
        CondensationEdge(source=u, target=v) for u, v in condensation.edges()
    ]
    edges.sort(key=lambda e: (e.source, e.target))

    return CondensationResult(
        vertex_count=len(sccs),
        components=components,
        edges=tuple(edges),
    )


def compute_acyclic_order(request: AcyclicOrderRequest) -> AcyclicOrderResult:
    """Compute a topological ordering of a directed acyclic graph.

    A cyclic graph is a typed ``acyclic=false`` outcome, not a host failure.
    """
    g = _build_digraph(request.graph)
    if not nx.is_directed_acyclic_graph(g):
        return AcyclicOrderResult(acyclic=False, order=())
    return AcyclicOrderResult(acyclic=True, order=tuple(nx.topological_sort(g)))
