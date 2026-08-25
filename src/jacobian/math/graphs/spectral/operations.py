"""Exact graph spectral operations backed by SymPy."""

from __future__ import annotations

from typing import Any

__all__ = [
    "adjacency_characteristic_polynomial",
    "adjacency_spectrum",
    "laplacian_characteristic_polynomial",
    "laplacian_spectrum",
]

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.spectral._models import _require_spectral_graph
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.polynomials.values import RationalPolynomial


def _characteristic_polynomial_coeffs(
    matrix: Any,
) -> tuple[CanonicalRational, ...]:
    """Return monic charpoly coefficients (increasing degree) as canonical values."""
    from fractions import Fraction

    coeffs = matrix.charpoly().all_coeffs()
    increasing = list(reversed(coeffs))
    return tuple(
        CanonicalRational.from_fraction(Fraction(int(c.p), int(c.q)))
        for c in increasing
    )


def _dense_to_canonical_polynomial(
    coefficients: tuple[CanonicalRational, ...],
) -> RationalPolynomial:
    from jacobian.math.graphs.spectral._models import _dense_to_canonical_polynomial

    return _dense_to_canonical_polynomial(coefficients)


def _adjacency_matrix(graph: IndexedSimpleUndirectedGraph) -> Any:
    import sympy

    _require_spectral_graph(graph)
    mat = sympy.zeros(graph.vertex_count)
    for u, v in graph.edges:
        mat[u, v] = 1
        mat[v, u] = 1
    return mat


def _laplacian_matrix(graph: IndexedSimpleUndirectedGraph) -> Any:
    import sympy

    adj = _adjacency_matrix(graph)
    degree = sympy.diag(*(sum(adj[vertex, :]) for vertex in range(graph.vertex_count)))
    return degree - adj


def adjacency_spectrum(graph: IndexedSimpleUndirectedGraph) -> list[tuple[str, int]]:
    _require_spectral_graph(graph)
    mat = _adjacency_matrix(graph)
    eigenvals = mat.eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]


def laplacian_spectrum(graph: IndexedSimpleUndirectedGraph) -> list[tuple[str, int]]:
    _require_spectral_graph(graph)
    eigenvals = _laplacian_matrix(graph).eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]


def adjacency_characteristic_polynomial(
    graph: IndexedSimpleUndirectedGraph,
) -> RationalPolynomial:
    """Return det(xI - A) as the canonical sparse rational polynomial."""

    return _dense_to_canonical_polynomial(
        _characteristic_polynomial_coeffs(_adjacency_matrix(graph))
    )


def laplacian_characteristic_polynomial(
    graph: IndexedSimpleUndirectedGraph,
) -> RationalPolynomial:
    """Return det(xI - L) as the canonical sparse rational polynomial."""

    return _dense_to_canonical_polynomial(
        _characteristic_polynomial_coeffs(_laplacian_matrix(graph))
    )
