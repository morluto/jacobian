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
        eigenvalues=tuple(v for v, _ in result),
        multiplicities=tuple(m for _, m in result),
    )


def compute_laplacian_spectrum(request: GraphSpectrumRequest) -> GraphSpectrumResult:
    result = laplacian_spectrum(request.graph)
    return GraphSpectrumResult(
        eigenvalues=tuple(v for v, _ in result),
        multiplicities=tuple(m for _, m in result),
    )


def compute_adjacency_characteristic_polynomial(
    request: GraphSpectrumRequest,
) -> GraphCharacteristicPolynomialResult:
    from fractions import Fraction

    from jacobian._exact import CanonicalRational

    coeffs = adjacency_characteristic_polynomial(request.graph)
    return GraphCharacteristicPolynomialResult(
        coefficients=tuple(
            CanonicalRational.from_fraction(Fraction(n, d)) for n, d in coeffs
        ),
    )


def compute_laplacian_characteristic_polynomial(
    request: GraphSpectrumRequest,
) -> GraphCharacteristicPolynomialResult:
    from fractions import Fraction

    from jacobian._exact import CanonicalRational

    coeffs = laplacian_characteristic_polynomial(request.graph)
    return GraphCharacteristicPolynomialResult(
        coefficients=tuple(
            CanonicalRational.from_fraction(Fraction(n, d)) for n, d in coeffs
        ),
    )
