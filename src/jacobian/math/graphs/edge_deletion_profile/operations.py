"""Edge-deletion chromatic profile kernel."""

from __future__ import annotations

from itertools import combinations

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.edge_deletion_profile._models import (
    DeletionEntry,
    EdgeDeletionProfileResult,
    _edge_deletion_admission_error,
    _is_bipartite,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_edge_deletion_profile"]


def _exact_chromatic_number(graph: nx.Graph[str]) -> int:
    """Compute exact chromatic number for small graphs via greedy + verification."""
    if graph.number_of_nodes() == 0:
        return 0
    if graph.number_of_edges() == 0:
        return 1

    # Try k from 1 upward using networkx greedy coloring as upper bound,
    # then verify by brute-force for small graphs
    coloring: dict[str, int] = nx.coloring.greedy_color(graph)
    upper = max(coloring.values()) + 1

    for k in range(1, upper + 1):
        if _k_colorable(graph, k):
            return k
    return upper


def _k_colorable(graph: nx.Graph[str], k: int) -> bool:
    """Check if graph is k-colorable using backtracking."""
    nodes = list(graph.nodes())
    if not nodes:
        return True
    colors: dict[str, int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(nodes):
            return True
        node = nodes[idx]
        used = set()
        for neighbor in graph.neighbors(node):
            if neighbor in colors:
                used.add(colors[neighbor])
        for c in range(k):
            if c not in used:
                colors[node] = c
                if backtrack(idx + 1):
                    return True
                del colors[node]
        return False

    return backtrack(0)


def compute_edge_deletion_profile(
    graph: SimpleUndirectedGraph,
    deletion_order: int,
) -> EdgeDeletionProfileResult:
    """Return the chromatic number of G-F for every edge subset F with |F| <= deletion_order."""
    failure = _edge_deletion_admission_error(graph, deletion_order)
    if failure is not None:
        code, message = failure
        raise OperationDomainValidationError(
            location=("graph", "deletion_order"),
            code=f"edge_deletion.{code}",
            message=message,
        )
    nx_graph: nx.Graph[str] = nx.Graph()
    for v in graph.vertices:
        nx_graph.add_node(v)
    for u, v in graph.edges:
        nx_graph.add_edge(u, v)

    edges_list = list(graph.edges)
    source_is_bipartite = _is_bipartite(graph)
    source_chi = (
        (1 if not graph.edges else 2)
        if source_is_bipartite
        else _exact_chromatic_number(nx_graph)
    )

    entries: list[DeletionEntry] = []

    for size in range(min(deletion_order, len(edges_list)) + 1):
        for subset in combinations(range(len(edges_list)), size):
            deleted_edges: list[tuple[str, str]] = []
            for i in subset:
                left, right = edges_list[i]
                deleted_edges.append((left, right))
            deleted = tuple(sorted(deleted_edges))
            if source_is_bipartite:
                chi = 1 if len(subset) == len(edges_list) else 2
            else:
                sub_graph: nx.Graph[str] = nx.Graph()
                for v in graph.vertices:
                    sub_graph.add_node(v)
                for idx, (u, v) in enumerate(graph.edges):
                    if idx not in subset:
                        sub_graph.add_edge(u, v)
                chi = _exact_chromatic_number(sub_graph)
            entries.append(
                DeletionEntry(
                    deleted_edges=deleted,
                    chromatic_number=chi,
                )
            )

    return EdgeDeletionProfileResult(
        graph=graph,
        source_chromatic_number=source_chi,
        deletion_order=deletion_order,
        entries=tuple(entries),
    )
