"""Exact graph spectral operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graph_spectral._models import (
    GraphSpectrumRequest,
    GraphSpectrumResult,
)
from jacobian.math.graph_spectral._operations import (
    compute_adjacency_spectrum,
    compute_laplacian_spectrum,
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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


GRAPH_SPECTRAL_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
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
)

TOOLS = GRAPH_SPECTRAL_OPERATIONS

__all__ = ["TOOLS"]
