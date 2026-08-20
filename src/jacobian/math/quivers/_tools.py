"""Quiver and path algebra operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.quivers._models import (
    AdjacencyMatricesRequest,
    AdjacencyMatricesResult,
    FixedLengthPathsRequest,
    FixedLengthPathsResult,
    VertexProfilesRequest,
    VertexProfilesResult,
)
from jacobian.math.quivers._operations import (
    compute_adjacency_matrices,
    compute_fixed_length_paths,
    compute_vertex_profiles,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "quiver.adjacency_matrices.compute",
        "Compute adjacency matrix and transpose of a quiver",
        "Compute the adjacency matrix and its transpose for a finite quiver.",
        AdjacencyMatricesRequest,
        AdjacencyMatricesResult,
        compute_adjacency_matrices,
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
        compute_vertex_profiles,
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
        compute_fixed_length_paths,
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
