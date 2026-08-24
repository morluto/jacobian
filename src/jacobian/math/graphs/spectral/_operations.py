"""Domain-owned graph spectral operations."""

from __future__ import annotations

from jacobian.math.graphs.spectral import (
    adjacency_characteristic_polynomial,
    adjacency_spectrum,
    laplacian_characteristic_polynomial,
    laplacian_spectrum,
)
from jacobian.math.graphs.spectral._models import (
    GraphCharacteristicPolynomialResult,
    GraphSpectrumRequest,
    GraphSpectrumResult,
)


def compute_adjacency_spectrum(request: GraphSpectrumRequest) -> GraphSpectrumResult:
    result = adjacency_spectrum(request.graph)
    return GraphSpectrumResult(
        graph=request.graph,
        matrix_convention="ADJACENCY",
        eigenvalues=tuple(v for v, _ in result),
        multiplicities=tuple(m for _, m in result),
    )


def compute_laplacian_spectrum(request: GraphSpectrumRequest) -> GraphSpectrumResult:
    result = laplacian_spectrum(request.graph)
    return GraphSpectrumResult(
        graph=request.graph,
        matrix_convention="LAPLACIAN",
        eigenvalues=tuple(v for v, _ in result),
        multiplicities=tuple(m for _, m in result),
    )


def compute_adjacency_characteristic_polynomial(
    request: GraphSpectrumRequest,
) -> GraphCharacteristicPolynomialResult:
    return GraphCharacteristicPolynomialResult(
        graph=request.graph,
        convention="ADJACENCY",
        polynomial=adjacency_characteristic_polynomial(request.graph),
    )


def compute_laplacian_characteristic_polynomial(
    request: GraphSpectrumRequest,
) -> GraphCharacteristicPolynomialResult:
    return GraphCharacteristicPolynomialResult(
        graph=request.graph,
        convention="LAPLACIAN",
        polynomial=laplacian_characteristic_polynomial(request.graph),
    )
