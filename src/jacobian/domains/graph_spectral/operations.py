"""Domain adapter for graph spectral operations."""

from __future__ import annotations

from jacobian.contracts.graph_spectral import (
    GraphCharacteristicPolynomialRequest,
    GraphCharacteristicPolynomialResult,
    GraphSpectrumRequest,
    GraphSpectrumResult,
)
from jacobian.math.graph_spectral import (
    adjacency_spectrum,
    characteristic_polynomial,
    laplacian_spectrum,
)


def compute_adjacency_spectrum(request: GraphSpectrumRequest) -> GraphSpectrumResult:
    result = adjacency_spectrum(  # type: ignore[no-untyped-call]
        request.graph.vertex_count,
        [list(e) for e in request.graph.edges],
    )
    return GraphSpectrumResult(
        eigenvalues=tuple(v for v, _ in result),
        multiplicities=tuple(m for _, m in result),
    )


def compute_laplacian_spectrum(request: GraphSpectrumRequest) -> GraphSpectrumResult:
    result = laplacian_spectrum(  # type: ignore[no-untyped-call]
        request.graph.vertex_count,
        [list(e) for e in request.graph.edges],
    )
    return GraphSpectrumResult(
        eigenvalues=tuple(v for v, _ in result),
        multiplicities=tuple(m for _, m in result),
    )


def compute_characteristic_polynomial(
    request: GraphCharacteristicPolynomialRequest,
) -> GraphCharacteristicPolynomialResult:
    degree, coeffs = characteristic_polynomial(  # type: ignore[no-untyped-call]
        request.graph.vertex_count,
        [list(e) for e in request.graph.edges],
        matrix=request.matrix,
    )
    return GraphCharacteristicPolynomialResult(
        degree=degree,
        coefficients_descending=coeffs,
        matrix=request.matrix,
    )
