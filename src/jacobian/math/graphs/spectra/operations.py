"""Exact graph spectral operations backed by SymPy."""

from __future__ import annotations

from typing import Any

from pydantic_core import PydanticCustomError

__all__ = [
    "adjacency_characteristic_polynomial",
    "adjacency_spectrum",
    "laplacian_characteristic_polynomial",
    "laplacian_spectrum",
]

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.spectra._models import (
    _require_characteristic_polynomial_graph,
    _require_spectral_graph,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.polynomials.values import RationalPolynomial


def _require_admitted_spectral_graph(
    graph: IndexedSimpleUndirectedGraph,
) -> None:
    try:
        _require_spectral_graph(graph)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("graph",),
            code=error.type,
            message=error.message(),
        ) from error


def _require_admitted_characteristic_polynomial_graph(
    graph: IndexedSimpleUndirectedGraph,
) -> None:
    try:
        _require_characteristic_polynomial_graph(graph)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("graph",),
            code=error.type,
            message=error.message(),
        ) from error


def _characteristic_polynomial_coeffs(
    entries: list[int], order: int
) -> tuple[CanonicalRational, ...]:
    """Return monic charpoly coefficients (increasing degree) as canonical values."""
    from flint import fmpz_mat

    coefficients = fmpz_mat(order, order, entries).charpoly().coeffs()
    return tuple(
        CanonicalRational(num=str(int(coefficient)), den="1")
        for coefficient in coefficients
    )


def _dense_to_canonical_polynomial(
    coefficients: tuple[CanonicalRational, ...],
) -> RationalPolynomial:
    from jacobian.math.graphs.spectra._models import _dense_to_canonical_polynomial

    return _dense_to_canonical_polynomial(coefficients)


def _adjacency_matrix(graph: IndexedSimpleUndirectedGraph) -> Any:
    import sympy

    _require_admitted_spectral_graph(graph)
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


def _adjacency_matrix_entries(graph: IndexedSimpleUndirectedGraph) -> list[int]:
    order = graph.vertex_count
    entries = [0] * (order * order)
    for left, right in graph.edges:
        entries[left * order + right] = 1
        entries[right * order + left] = 1
    return entries


def _laplacian_matrix_entries(graph: IndexedSimpleUndirectedGraph) -> list[int]:
    order = graph.vertex_count
    entries = [0] * (order * order)
    degrees = [0] * order
    for left, right in graph.edges:
        entries[left * order + right] = -1
        entries[right * order + left] = -1
        degrees[left] += 1
        degrees[right] += 1
    for vertex, degree in enumerate(degrees):
        entries[vertex * order + vertex] = degree
    return entries


def adjacency_spectrum(graph: IndexedSimpleUndirectedGraph) -> list[tuple[str, int]]:
    mat = _adjacency_matrix(graph)
    eigenvals = mat.eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]


def laplacian_spectrum(graph: IndexedSimpleUndirectedGraph) -> list[tuple[str, int]]:
    eigenvals = _laplacian_matrix(graph).eigenvals()
    return [(str(val), int(mult)) for val, mult in eigenvals.items()]


def adjacency_characteristic_polynomial(
    graph: IndexedSimpleUndirectedGraph,
) -> RationalPolynomial:
    """Return det(xI - A) as the canonical sparse rational polynomial."""

    _require_admitted_characteristic_polynomial_graph(graph)
    order = graph.vertex_count
    return _dense_to_canonical_polynomial(
        _characteristic_polynomial_coeffs(_adjacency_matrix_entries(graph), order)
    )


def laplacian_characteristic_polynomial(
    graph: IndexedSimpleUndirectedGraph,
) -> RationalPolynomial:
    """Return det(xI - L) as the canonical sparse rational polynomial."""

    _require_admitted_characteristic_polynomial_graph(graph)
    order = graph.vertex_count
    return _dense_to_canonical_polynomial(
        _characteristic_polynomial_coeffs(_laplacian_matrix_entries(graph), order)
    )
