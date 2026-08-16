"""Exact graph spectral operations backed by SymPy."""

from __future__ import annotations

__all__ = ["adjacency_spectrum", "laplacian_spectrum"]


def _adjacency_matrix(vertex_count, edges):
    import sympy

    mat = sympy.zeros(vertex_count)
    for u, v in edges:
        mat[u, v] = 1
        mat[v, u] = 1
    return mat


def adjacency_spectrum(vertex_count, edges):
    if not 1 <= vertex_count <= 32:
        raise ValueError("graph vertex count must be between 1 and 32")
    mat = _adjacency_matrix(vertex_count, edges)
    eigenvals = mat.eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]


def laplacian_spectrum(vertex_count, edges):
    import sympy

    if not 1 <= vertex_count <= 32:
        raise ValueError("graph vertex count must be between 1 and 32")
    adj = _adjacency_matrix(vertex_count, edges)
    degree = sympy.zeros(vertex_count)
    for u, v in edges:
        degree[u, u] += 1
        degree[v, v] += 1
    lap = degree - adj
    eigenvals = lap.eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]
