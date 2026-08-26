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
    return GraphSpectrumResult._from_kernel(
        graph=request.graph,
        matrix_convention="ADJACENCY",
        eigenvalues=tuple(v for v, _ in result),
        multiplicities=tuple(m for _, m in result),
    )


def compute_laplacian_spectrum(request: GraphSpectrumRequest) -> GraphSpectrumResult:
    result = laplacian_spectrum(request.graph)
    return GraphSpectrumResult._from_kernel(
        graph=request.graph,
        matrix_convention="LAPLACIAN",
        eigenvalues=tuple(v for v, _ in result),
        multiplicities=tuple(m for _, m in result),
    )


def compute_adjacency_characteristic_polynomial(
    request: GraphSpectrumRequest,
) -> GraphCharacteristicPolynomialResult:
    return GraphCharacteristicPolynomialResult._from_kernel(
        graph=request.graph,
        convention="ADJACENCY",
        polynomial=adjacency_characteristic_polynomial(request.graph),
    )


def compute_laplacian_characteristic_polynomial(
    request: GraphSpectrumRequest,
) -> GraphCharacteristicPolynomialResult:
    return GraphCharacteristicPolynomialResult._from_kernel(
        graph=request.graph,
        convention="LAPLACIAN",
        polynomial=laplacian_characteristic_polynomial(request.graph),
    )


def verify_graph_spectrum_result(result: GraphSpectrumResult) -> bool:
    """Verify a claimed exact spectrum inside the spectral graph envelope."""

    if (
        len(result.eigenvalues) != len(result.multiplicities)
        or len(set(result.eigenvalues)) != len(result.eigenvalues)
        or any(multiplicity < 1 for multiplicity in result.multiplicities)
        or sum(result.multiplicities) != result.graph.vertex_count
    ):
        return False
    expected = (
        adjacency_spectrum(result.graph)
        if result.matrix_convention == "ADJACENCY"
        else laplacian_spectrum(result.graph)
    )
    return dict(expected) == dict(
        zip(result.eigenvalues, result.multiplicities, strict=True)
    )


def verify_graph_characteristic_polynomial_result(
    result: GraphCharacteristicPolynomialResult,
) -> bool:
    """Verify a claimed graph characteristic polynomial in its admitted envelope."""

    if result.polynomial.variables != ("x",):
        return False
    expected = (
        adjacency_characteristic_polynomial(result.graph)
        if result.convention == "ADJACENCY"
        else laplacian_characteristic_polynomial(result.graph)
    )
    return result.polynomial == expected
