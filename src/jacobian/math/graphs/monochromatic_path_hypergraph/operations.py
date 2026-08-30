"""Monochromatic path hypergraph kernel."""

from __future__ import annotations

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.monochromatic_path_hypergraph._models import (
    MonochromaticPathHypergraphResult,
    MonochromaticPathResult,
    _monochromatic_path_admission_error,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph

__all__ = ["construct_monochromatic_path_hypergraphs"]


def construct_monochromatic_path_hypergraphs(
    colored_graph: ColoredUndirectedGraph,
) -> MonochromaticPathHypergraphResult:
    """For each colour, return a FiniteHypergraph whose edges are the vertex
    supports of simple paths using only that colour.

    A singleton vertex is included in every colour's hypergraph.
    """
    failure = _monochromatic_path_admission_error(colored_graph)
    if failure is not None:
        code, message = failure
        raise OperationDomainValidationError(
            location=("graph",), code=f"monochromatic_path.{code}", message=message
        )
    graph = colored_graph.graph
    edge_colors = colored_graph.edge_colors
    vertices = graph.vertices
    edges = graph.edges

    # Group edges by color
    color_edges: dict[str, list[tuple[str, str]]] = {}
    if edge_colors:
        for idx, color in enumerate(edge_colors):
            color_edges.setdefault(color, []).append(edges[idx])
    else:
        # No edge colors - single "default" color with all edges
        for e in edges:
            color_edges.setdefault("uncolored", []).append(e)

    per_color: list[MonochromaticPathResult] = []

    for color in sorted(color_edges.keys()):
        edges_list = color_edges[color]

        nx_graph: nx.Graph[str] = nx.Graph()
        for v in vertices:
            nx_graph.add_node(v)
        for u, v in edges_list:
            nx_graph.add_edge(u, v)

        vertex_list = list(vertices)
        n = len(vertex_list)

        supports: list[tuple[str, ...]] = [(v,) for v in vertex_list]

        from itertools import combinations

        for size in range(2, n + 1):
            for subset in combinations(vertex_list, size):
                sub_nx = nx_graph.subgraph(subset)
                if not nx.is_connected(sub_nx):
                    continue
                if _has_hamiltonian_path(sub_nx):
                    supports.append(tuple(sorted(subset)))

        edges_hg = tuple((f"e_{i}", tuple(s)) for i, s in enumerate(supports))
        hypergraph = FiniteHypergraph(
            vertices=vertices,
            edges=edges_hg,
        )

        per_color.append(
            MonochromaticPathResult(
                color=color,
                hypergraph=hypergraph,
            )
        )

    return MonochromaticPathHypergraphResult(
        graph=colored_graph,
        per_color=tuple(per_color),
    )


def _has_hamiltonian_path(graph: nx.Graph[str]) -> bool:
    """Check if a graph has a Hamiltonian path."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if n <= 1:
        return True
    if n == 2:
        return graph.has_edge(nodes[0], nodes[1])

    for start in nodes:
        visited = {start}
        path = [start]

        if _search_hamiltonian_path(graph, n, path, visited):
            return True

    return False


def _search_hamiltonian_path(
    graph: nx.Graph[str],
    vertex_count: int,
    path: list[str],
    visited: set[str],
) -> bool:
    if len(path) == vertex_count:
        return True
    current = path[-1]
    for neighbor in graph.neighbors(current):
        if neighbor not in visited:
            visited.add(neighbor)
            path.append(neighbor)
            if _search_hamiltonian_path(graph, vertex_count, path, visited):
                return True
            path.pop()
            visited.remove(neighbor)
    return False
