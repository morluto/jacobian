"""Exact graph spectral operations backed by SymPy."""

from __future__ import annotations

from typing import Any

__all__ = [
    "adjacency_characteristic_polynomial",
    "adjacency_spectrum",
    "laplacian_characteristic_polynomial",
    "laplacian_spectrum",
]

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


def _characteristic_polynomial_coeffs(matrix: Any) -> list[tuple[int, int]]:
    """Return monic charpoly coefficients (increasing degree) as (num, den)."""
    from fractions import Fraction

    poly = matrix.charpoly()
    # sympy Poly.all_coeffs() is decreasing degree; reverse for increasing.
    coeffs = poly.all_coeffs()
    increasing = list(reversed(coeffs))
    result: list[tuple[int, int]] = []
    for c in increasing:
        r = Fraction(int(c.p), int(c.q))
        result.append((r.numerator, r.denominator))
    return result


def adjacency_characteristic_polynomial(graph: GraphEdgeList) -> list[tuple[int, int]]:
    """Monic characteristic polynomial of the adjacency matrix (increasing degree)."""
    return _characteristic_polynomial_coeffs(_adjacency_matrix(graph))


def laplacian_characteristic_polynomial(graph: GraphEdgeList) -> list[tuple[int, int]]:
    """Monic characteristic polynomial of the Laplacian matrix (increasing degree)."""
    import sympy

    adj = _adjacency_matrix(graph)
    degree = sympy.diag(*(sum(adj[vertex, :]) for vertex in range(graph.vertex_count)))
    lap = degree - adj
    return _characteristic_polynomial_coeffs(lap)
