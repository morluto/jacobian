"""Quiver and path algebra operation declarations."""

from typing import Any

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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="quiver.adjacency_matrices.compute",
        title="Compute adjacency matrix and transpose of a quiver",
        description="Compute the adjacency matrix and its transpose for a finite quiver.",
        request_type=AdjacencyMatricesRequest,
        result_type=AdjacencyMatricesResult,
        run=_adjacency_matrices,
        tags=("quiver", "adjacency", "exact"),
        examples=(
            OperationExample(
                name="kronecker_quiver",
                description="Compute adjacency matrices of the Kronecker quiver.",
                input={
                    "quiver": {
                        "vertex_count": 2,
                        "arrows": [[0, 1], [0, 1]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quiver.vertex_profiles.compute",
        title="Compute in-degree and out-degree profiles of a quiver",
        description="Compute the in-degree and out-degree for each vertex of a finite quiver.",
        request_type=VertexProfilesRequest,
        result_type=VertexProfilesResult,
        run=_vertex_profiles,
        tags=("quiver", "vertex-profiles", "exact"),
        examples=(
            OperationExample(
                name="kronecker_quiver",
                description="Compute vertex profiles of the Kronecker quiver.",
                input={
                    "quiver": {
                        "vertex_count": 2,
                        "arrows": [[0, 1], [0, 1]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quiver.paths.fixed_length.compute",
        title="Count paths of fixed length in a quiver",
        description="Count the number of paths of a fixed length between all vertex "
        "pairs using adjacency matrix powers.",
        request_type=FixedLengthPathsRequest,
        result_type=FixedLengthPathsResult,
        run=_fixed_length_paths,
        tags=("quiver", "paths", "exact"),
        examples=(
            OperationExample(
                name="path_count",
                description="Count length-2 paths in a triangle quiver.",
                input={
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
