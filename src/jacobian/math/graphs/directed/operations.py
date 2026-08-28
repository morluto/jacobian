"""Domain-owned directed graph operations."""

from __future__ import annotations

from typing import Any

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.directed._models import (
    MAX_DIRECTED_OPERATION_EDGES,
    MAX_DIRECTED_OPERATION_VERTICES,
    AcyclicOrderResult,
    CondensationEdge,
    CondensationResult,
    DagLongestPathResult,
    DirectedGraph,
    ReachabilityResult,
    StronglyConnectedComponentsResult,
)


def _build_digraph(graph: DirectedGraph) -> nx.DiGraph[int]:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(graph.vertex_count))
    for source, target in graph.edges:
        g.add_edge(source, target)
    return g


def _admit_directed_graph(graph: DirectedGraph) -> None:
    if graph.vertex_count > MAX_DIRECTED_OPERATION_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertex_count"),
            code="graph.directed_vertex_budget_exceeded",
            message=(
                "directed graph operation supports at most "
                f"{MAX_DIRECTED_OPERATION_VERTICES} vertices"
            ),
        )
    if len(graph.edges) > MAX_DIRECTED_OPERATION_EDGES:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.directed_edge_budget_exceeded",
            message=(
                "directed graph operation supports at most "
                f"{MAX_DIRECTED_OPERATION_EDGES} edges"
            ),
        )


def reachability(graph: DirectedGraph, source: int) -> ReachabilityResult:
    """Determine which vertices are reachable from the source vertex.

    A vertex is reachable if there is a directed path from source to that
    vertex. The source itself is always considered reachable.
    """
    _admit_directed_graph(graph)
    if not 0 <= source < graph.vertex_count:
        raise OperationDomainValidationError(
            location=("source",),
            code="graph.source_must_be_in_0_graph_vertex_count_1",
            message="source must be in 0..graph.vertex_count-1",
        )
    g = _build_digraph(graph)
    descendants = nx.descendants(g, source)
    reachable = frozenset(descendants) | {source}
    unreachable = frozenset(range(graph.vertex_count)) - reachable
    return ReachabilityResult(
        source=source,
        reachable=tuple(sorted(reachable)),
        unreachable=tuple(sorted(unreachable)),
    )


def strongly_connected_components(
    graph: DirectedGraph,
) -> StronglyConnectedComponentsResult:
    """Partition the graph into strongly connected components.

    Components are returned in the order NetworkX yields them; each
    component's vertices are sorted for determinism.
    """
    _admit_directed_graph(graph)
    g = _build_digraph(graph)
    sccs = list(nx.strongly_connected_components(g))
    components = tuple(tuple(sorted(component)) for component in sccs)
    return StronglyConnectedComponentsResult(
        component_count=len(components),
        components=components,
    )


def condensation(graph: DirectedGraph) -> CondensationResult:
    """Compute the condensation of the graph.

    The condensation is the DAG whose vertices are the strongly connected
    components of the original graph. Condensation vertex ``i`` corresponds to
    the ``i``-th strongly connected component returned by NetworkX (and
    reported in the ``components`` field).
    """
    _admit_directed_graph(graph)
    g = _build_digraph(graph)
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


def acyclic_order(graph: DirectedGraph) -> AcyclicOrderResult:
    """Compute a topological ordering of a directed acyclic graph.

    A cyclic graph is a typed ``acyclic=false`` outcome, not a host failure.
    """
    _admit_directed_graph(graph)
    g = _build_digraph(graph)
    if not nx.is_directed_acyclic_graph(g):
        return AcyclicOrderResult(acyclic=False, order=())
    return AcyclicOrderResult(acyclic=True, order=tuple(nx.topological_sort(g)))


def dag_longest_path(graph: DirectedGraph) -> DagLongestPathResult:
    """Compute the exact longest directed simple path in a DAG.

    A cyclic graph yields a typed ``NOT_APPLICABLE`` outcome, not a host
    error.  For a DAG the exact maximum edge count and a canonical path
    witness are returned.  Ties between maximizers are broken by choosing
    the lexicographically least path vertex sequence.
    """
    _admit_directed_graph(graph)
    g = _build_digraph(graph)
    if not nx.is_directed_acyclic_graph(g):
        return DagLongestPathResult(
            status="NOT_APPLICABLE",
            source=graph,
        )
    topo_order = list(nx.topological_sort(g))

    # longest_from[v] = (edge_count, path_from_v) for the best path starting
    # at vertex v.  Iterate in reverse topological order so all successors
    # are resolved before their predecessors.
    longest_from: dict[int, tuple[int, list[int]]] = {}
    for v in reversed(topo_order):
        best_edges = 0
        best_path: list[int] = [v]
        for w in g.successors(v):
            cand_edges, cand_path = longest_from[w]
            if cand_edges + 1 > best_edges or (
                cand_edges + 1 == best_edges and [v, *cand_path] < best_path
            ):
                best_edges = cand_edges + 1
                best_path = [v, *cand_path]
        longest_from[v] = (best_edges, best_path)

    # Find the global maximum, breaking ties by the lexicographically least
    # vertex sequence.
    best_edge_count = -1
    best_vertex_sequence: list[int] = []
    for v in topo_order:
        cand_edges, cand_path = longest_from[v]
        if cand_edges > best_edge_count or (
            cand_edges == best_edge_count and cand_path < best_vertex_sequence
        ):
            best_edge_count = cand_edges
            best_vertex_sequence = cand_path

    return DagLongestPathResult(
        status="ACYCLIC",
        maximum_edge_count=best_edge_count,
        path=tuple(best_vertex_sequence),
        source=graph,
    )


__all__ = [
    "acyclic_order",
    "condensation",
    "dag_longest_path",
    "reachability",
    "strongly_connected_components",
]
