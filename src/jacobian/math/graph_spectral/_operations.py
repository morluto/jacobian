"""Domain-owned graph spectral operations."""

from __future__ import annotations

from jacobian.math.graph_spectral import adjacency_spectrum, laplacian_spectrum
from jacobian.math.graph_spectral._models import (
    GraphSpectrumRequest,
    GraphSpectrumResult,
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
