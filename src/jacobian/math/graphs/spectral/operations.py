"""Exact graph spectral operations backed by SymPy."""

from __future__ import annotations

from typing import Any

__all__ = ["adjacency_spectrum", "laplacian_spectrum"]

from jacobian.math.graphs.spectral._models import GraphEdgeList


def _adjacency_matrix(graph: GraphEdgeList) -> Any:
    import sympy

    mat = sympy.zeros(graph.vertex_count)
    for u, v in graph.edges:
        mat[u, v] = 1
        mat[v, u] = 1
    return mat


def adjacency_spectrum(graph: GraphEdgeList) -> list[tuple[str, int]]:
    mat = _adjacency_matrix(graph)
    eigenvals = mat.eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]


def laplacian_spectrum(graph: GraphEdgeList) -> list[tuple[str, int]]:
    import sympy

    adj = _adjacency_matrix(graph)
    degree = sympy.diag(*(sum(adj[vertex, :]) for vertex in range(graph.vertex_count)))
    lap = degree - adj
    eigenvals = lap.eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]
