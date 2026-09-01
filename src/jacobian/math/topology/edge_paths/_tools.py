"""Algebraic topology operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.edge_paths._models import (
    EdgePathConcatenateRequest,
    EdgePathConcatenateResult,
    EdgePathWordRequest,
    EdgePathWordResult,
)
from jacobian.math.topology.edge_paths.operations import (
    concatenate_edge_paths,
    edge_path_word,
)


def _word(request: EdgePathWordRequest) -> EdgePathWordResult:
    return edge_path_word(
        request.vertex_count, request.edges, request.start_vertex, request.path
    )


def _concatenate(request: EdgePathConcatenateRequest) -> EdgePathConcatenateResult:
    return concatenate_edge_paths(request.vertex_count, request.path_a, request.path_b)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="topology.simplicial.edge_path.word.compute",
        title="Compute the free group word for an edge path",
        description="Compute the free group word representation of an edge path in a "
        "graph, where each edge corresponds to a generator and its inverse.",
        request_type=EdgePathWordRequest,
        result_type=EdgePathWordResult,
        run=_word,
        tags=("topology", "edge-path", "exact"),
        examples=(
            OperationExample(
                name="triangle_path",
                description="Compute the word for path 0->1->2 in a triangle.",
                input={
                    "vertex_count": 3,
                    "edges": [[0, 1], [1, 2], [2, 0]],
                    "start_vertex": 0,
                    "path": [
                        {"edge_index": 0, "orientation": 1},
                        {"edge_index": 1, "orientation": 1},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial.edge_path.concatenate.compute",
        title="Concatenate two edge paths",
        description="Concatenate two edge paths in a graph, removing the shared vertex.",
        request_type=EdgePathConcatenateRequest,
        result_type=EdgePathConcatenateResult,
        run=_concatenate,
        tags=("topology", "edge-path", "exact"),
        examples=(
            OperationExample(
                name="concatenate_paths",
                description="Concatenate [0,1] and [1,2] in a 3-vertex graph.",
                input={
                    "vertex_count": 3,
                    "path_a": [0, 1],
                    "path_b": [1, 2],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
