"""Exact graph spectral operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def graph_spectral_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


PATH_P3 = {"graph": {"vertex_count": 3, "edges": [[0, 1], [1, 2]]}}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    graph_spectral_operation(
        "graph.spectrum.adjacency.compute",
        "Compute exact adjacency matrix eigenvalues",
        "Compute the exact eigenvalues with algebraic multiplicities of the adjacency matrix of a simple undirected graph using SymPy.",
        GraphSpectrumRequest,
        GraphSpectrumResult,
        compute_adjacency_spectrum,
        "graph",
        "spectrum",
        "adjacency",
        "eigenvalues",
        "exact",
        examples=(
            example(
                "path_adjacency_spectrum",
                "Compute the adjacency spectrum of a path graph P3.",
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2]],
                    }
                },
            ),
        ),
    ),
    graph_spectral_operation(
        "graph.spectrum.laplacian.compute",
        "Compute exact Laplacian matrix eigenvalues",
        "Compute the exact eigenvalues with algebraic multiplicities of the Laplacian matrix of a simple undirected graph using SymPy.",
        GraphSpectrumRequest,
        GraphSpectrumResult,
        compute_laplacian_spectrum,
        "graph",
        "spectrum",
        "laplacian",
        "eigenvalues",
        "exact",
        examples=(
            example(
                "path_laplacian_spectrum",
                "Compute the Laplacian spectrum of a path graph P3.",
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2]],
                    }
                },
            ),
        ),
    ),
    graph_spectral_operation(
        "graph.spectrum.adjacency.characteristic_polynomial.compute",
        "Compute exact adjacency characteristic polynomial",
        "Compute the exact monic characteristic polynomial det(xI - A) of the "
        "adjacency matrix of a simple undirected graph over QQ, returned as "
        "the canonical sparse RationalPolynomial in x with nonzero terms "
        "serialized in descending exponent order, using FLINT.",
        GraphCharacteristicPolynomialRequest,
        GraphCharacteristicPolynomialResult,
        compute_adjacency_characteristic_polynomial,
        "graph",
        "spectrum",
        "adjacency",
        "characteristic-polynomial",
        "exact",
        examples=(
            example(
                "path_adjacency_charpoly",
                (
                    "Characteristic polynomial of the adjacency matrix of P3; "
                    "the graph must be simple with at most 256 vertices."
                ),
                PATH_P3,
            ),
        ),
    ),
    graph_spectral_operation(
        "graph.spectrum.laplacian.characteristic_polynomial.compute",
        "Compute exact Laplacian characteristic polynomial",
        "Compute the exact monic characteristic polynomial det(xI - L) of the "
        "Laplacian matrix of a simple undirected graph over QQ, returned as "
        "the canonical sparse RationalPolynomial in x with nonzero terms "
        "serialized in descending exponent order, using FLINT.",
        GraphCharacteristicPolynomialRequest,
        GraphCharacteristicPolynomialResult,
        compute_laplacian_characteristic_polynomial,
        "graph",
        "spectrum",
        "laplacian",
        "characteristic-polynomial",
        "exact",
        examples=(
            example(
                "path_laplacian_charpoly",
                (
                    "Characteristic polynomial of the Laplacian matrix of P3; "
                    "the graph must be simple with at most 256 vertices."
                ),
                PATH_P3,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
