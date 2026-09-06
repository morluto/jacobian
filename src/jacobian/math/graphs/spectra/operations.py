"""Exact graph spectral operations backed by SymPy."""

from __future__ import annotations

from typing import Any

from pydantic_core import PydanticCustomError

__all__ = [
    "adjacency_characteristic_polynomial",
    "adjacency_spectrum",
    "laplacian_characteristic_polynomial",
    "laplacian_spectrum",
    "verify_spectrum",
]

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.spectra._models import (
    GraphSpectrumEntry,
    GraphSpectrumResult,
    _require_characteristic_polynomial_graph,
    _require_spectral_graph,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.number_theory.algebraic_numbers.real import (
    RealAlgebraicValue,
    isolate_real_algebraic,
)
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
        CanonicalRational(num=int(coefficient), den=1) for coefficient in coefficients
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


def _algebraic_value(value: Any) -> RealAlgebraicValue:
    import sympy

    x = sympy.Symbol("x")
    polynomial = sympy.Poly(sympy.minimal_polynomial(value, x), x)
    roots = polynomial.all_roots()
    try:
        root_index = roots.index(value)
    except ValueError as exc:
        raise ValueError("spectrum root is not on its minimal-polynomial axis") from exc
    coefficients = tuple(
        str(int(coefficient)) for coefficient in polynomial.all_coeffs()
    )
    return RealAlgebraicValue._from_admitted_polynomial(
        polynomial=coefficients,
        real_root_index=root_index,
    )


def _spectrum(matrix: Any) -> tuple[GraphSpectrumEntry, ...]:
    return tuple(
        GraphSpectrumEntry(
            value=_algebraic_value(value), multiplicity=int(multiplicity)
        )
        for value, multiplicity in matrix.eigenvals().items()
    )


def adjacency_spectrum(
    graph: IndexedSimpleUndirectedGraph,
) -> tuple[GraphSpectrumEntry, ...]:
    mat = _adjacency_matrix(graph)
    return _spectrum(mat)


def laplacian_spectrum(
    graph: IndexedSimpleUndirectedGraph,
) -> tuple[GraphSpectrumEntry, ...]:
    return _spectrum(_laplacian_matrix(graph))


def verify_spectrum(claim: GraphSpectrumResult) -> bool:
    """Check typed algebraic eigenvalue claims against the retained graph."""
    try:
        for entry in claim.spectrum:
            isolate_real_algebraic(entry.value)
        actual = (
            adjacency_spectrum(claim.graph)
            if claim.matrix_convention == "ADJACENCY"
            else laplacian_spectrum(claim.graph)
        )

        def key(entry: GraphSpectrumEntry) -> tuple[tuple[str, ...], int]:
            return entry.value.polynomial, entry.value.real_root_index

        return sorted((key(entry), entry.multiplicity) for entry in actual) == sorted(
            (key(entry), entry.multiplicity) for entry in claim.spectrum
        )
    except (
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        OperationDomainValidationError,
    ):
        return False


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
