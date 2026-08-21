"""Domain-owned graph spectral operations."""

from __future__ import annotations

from jacobian.math.graphs.spectral import adjacency_spectrum, laplacian_spectrum
from jacobian.math.graphs.spectral._models import (
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
