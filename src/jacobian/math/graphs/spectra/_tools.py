"""Exact graph spectral operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.spectra import (
    adjacency_characteristic_polynomial,
    adjacency_spectrum,
    laplacian_characteristic_polynomial,
    laplacian_spectrum,
)
from jacobian.math.graphs.spectra._models import (
    GraphCharacteristicPolynomialRequest,
    GraphCharacteristicPolynomialResult,
    GraphSpectrumRequest,
    GraphSpectrumResult,
)


def compute_adjacency_spectrum(request: GraphSpectrumRequest) -> GraphSpectrumResult:
    result = adjacency_spectrum(request.graph)
    return GraphSpectrumResult._from_kernel(
        graph=request.graph,
        matrix_convention="ADJACENCY",
        spectrum=result,
    )


def compute_laplacian_spectrum(request: GraphSpectrumRequest) -> GraphSpectrumResult:
    result = laplacian_spectrum(request.graph)
    return GraphSpectrumResult._from_kernel(
        graph=request.graph,
        matrix_convention="LAPLACIAN",
        spectrum=result,
    )


def compute_adjacency_characteristic_polynomial(
    request: GraphCharacteristicPolynomialRequest,
) -> GraphCharacteristicPolynomialResult:
    return GraphCharacteristicPolynomialResult._from_kernel(
        graph=request.graph,
        convention="ADJACENCY",
        polynomial=adjacency_characteristic_polynomial(request.graph),
    )


def compute_laplacian_characteristic_polynomial(
    request: GraphCharacteristicPolynomialRequest,
) -> GraphCharacteristicPolynomialResult:
    return GraphCharacteristicPolynomialResult._from_kernel(
        graph=request.graph,
        convention="LAPLACIAN",
        polynomial=laplacian_characteristic_polynomial(request.graph),
    )


PATH_P3 = {"graph": {"vertex_count": 3, "edges": [[0, 1], [1, 2]]}}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.spectrum.adjacency.compute",
        title="Compute exact adjacency matrix eigenvalues",
        description="Compute the exact eigenvalues with algebraic multiplicities of the adjacency matrix of a simple undirected graph using SymPy.",
        request_type=GraphSpectrumRequest,
        result_type=GraphSpectrumResult,
        run=compute_adjacency_spectrum,
        tags=("graph", "spectrum", "adjacency", "eigenvalues", "exact"),
        examples=(
            OperationExample(
                name="path_adjacency_spectrum",
                description="Compute the adjacency spectrum of a path graph P3.",
                input={
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.spectrum.laplacian.compute",
        title="Compute exact Laplacian matrix eigenvalues",
        description="Compute the exact eigenvalues with algebraic multiplicities of the Laplacian matrix of a simple undirected graph using SymPy.",
        request_type=GraphSpectrumRequest,
        result_type=GraphSpectrumResult,
        run=compute_laplacian_spectrum,
        tags=("graph", "spectrum", "laplacian", "eigenvalues", "exact"),
        examples=(
            OperationExample(
                name="path_laplacian_spectrum",
                description="Compute the Laplacian spectrum of a path graph P3.",
                input={
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.spectrum.adjacency.characteristic_polynomial.compute",
        title="Compute exact adjacency characteristic polynomial",
        description="Compute the exact monic characteristic polynomial det(xI - A) of the "
        "adjacency matrix of a simple undirected graph over QQ, returned as "
        "the canonical sparse RationalPolynomial in x with nonzero terms "
        "serialized in descending exponent order, using FLINT.",
        request_type=GraphCharacteristicPolynomialRequest,
        result_type=GraphCharacteristicPolynomialResult,
        run=compute_adjacency_characteristic_polynomial,
        tags=("graph", "spectrum", "adjacency", "characteristic-polynomial", "exact"),
        examples=(
            OperationExample(
                name="path_adjacency_charpoly",
                description=(
                    "Characteristic polynomial of the adjacency matrix of P3; "
                    "the graph must be simple with at most 256 vertices."
                ),
                input=PATH_P3,
            ),
        ),
    ),
    MathTool(
        operation_id="graph.spectrum.laplacian.characteristic_polynomial.compute",
        title="Compute exact Laplacian characteristic polynomial",
        description="Compute the exact monic characteristic polynomial det(xI - L) of the "
        "Laplacian matrix of a simple undirected graph over QQ, returned as "
        "the canonical sparse RationalPolynomial in x with nonzero terms "
        "serialized in descending exponent order, using FLINT.",
        request_type=GraphCharacteristicPolynomialRequest,
        result_type=GraphCharacteristicPolynomialResult,
        run=compute_laplacian_characteristic_polynomial,
        tags=("graph", "spectrum", "laplacian", "characteristic-polynomial", "exact"),
        examples=(
            OperationExample(
                name="path_laplacian_charpoly",
                description=(
                    "Characteristic polynomial of the Laplacian matrix of P3; "
                    "the graph must be simple with at most 256 vertices."
                ),
                input=PATH_P3,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
