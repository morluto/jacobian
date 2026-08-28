"""Quiver and path algebra operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.quivers._models import (
    AdjacencyMatricesRequest,
    AdjacencyMatricesResult,
    FixedLengthPathsRequest,
    FixedLengthPathsResult,
    VertexProfilesRequest,
    VertexProfilesResult,
)
from jacobian.math.graphs.quivers.operations import (
    adjacency_matrices,
    fixed_length_paths,
    vertex_profiles,
)


def _adjacency_matrices(request: AdjacencyMatricesRequest) -> AdjacencyMatricesResult:
    return adjacency_matrices(request.quiver)


def _vertex_profiles(request: VertexProfilesRequest) -> VertexProfilesResult:
    return vertex_profiles(request.quiver)


def _fixed_length_paths(request: FixedLengthPathsRequest) -> FixedLengthPathsResult:
    return fixed_length_paths(request.quiver, request.length)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "quiver.adjacency_matrices.compute",
        "Compute adjacency matrix and transpose of a quiver",
        "Compute the adjacency matrix and its transpose for a finite quiver.",
        AdjacencyMatricesRequest,
        AdjacencyMatricesResult,
        _adjacency_matrices,
        "quiver",
        "adjacency",
        "exact",
        examples=(
            example(
                "kronecker_quiver",
                "Compute adjacency matrices of the Kronecker quiver.",
                {
                    "quiver": {
                        "vertex_count": 2,
                        "arrows": [[0, 1], [0, 1]],
                    },
                },
            ),
        ),
    ),
    _op(
        "quiver.vertex_profiles.compute",
        "Compute in-degree and out-degree profiles of a quiver",
        "Compute the in-degree and out-degree for each vertex of a finite quiver.",
        VertexProfilesRequest,
        VertexProfilesResult,
        _vertex_profiles,
        "quiver",
        "vertex-profiles",
        "exact",
        examples=(
            example(
                "kronecker_quiver",
                "Compute vertex profiles of the Kronecker quiver.",
                {
                    "quiver": {
                        "vertex_count": 2,
                        "arrows": [[0, 1], [0, 1]],
                    },
                },
            ),
        ),
    ),
    _op(
        "quiver.paths.fixed_length.compute",
        "Count paths of fixed length in a quiver",
        "Count the number of paths of a fixed length between all vertex "
        "pairs using adjacency matrix powers.",
        FixedLengthPathsRequest,
        FixedLengthPathsResult,
        _fixed_length_paths,
        "quiver",
        "paths",
        "exact",
        examples=(
            example(
                "path_count",
                "Count length-2 paths in a triangle quiver.",
                {
                    "quiver": {
                        "vertex_count": 3,
                        "arrows": [[0, 1], [1, 2], [2, 0]],
                    },
                    "length": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
