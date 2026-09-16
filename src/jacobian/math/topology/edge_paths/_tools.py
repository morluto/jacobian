"""Algebraic topology operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.edge_paths._models import (
    EdgePathConcatenateRequest,
    EdgePathConcatenateResult,
    EdgePathWordRequest,
    EdgePathWordResult,
    FundamentalGroupPresentationRequest,
    FundamentalGroupPresentationResult,
)
from jacobian.math.topology.edge_paths.operations import (
    concatenate_edge_paths,
    edge_path_word,
    fundamental_group_presentation,
)


def _word(request: EdgePathWordRequest) -> EdgePathWordResult:
    return edge_path_word(
        request.vertex_count, request.edges, request.start_vertex, request.path
    )


def _concatenate(request: EdgePathConcatenateRequest) -> EdgePathConcatenateResult:
    return concatenate_edge_paths(request.vertex_count, request.path_a, request.path_b)


def _fundamental_group(
    request: FundamentalGroupPresentationRequest,
) -> FundamentalGroupPresentationResult:
    return fundamental_group_presentation(request)


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
    MathTool(
        operation_id="topology.simplicial.fundamental_group.presentation.compute",
        title="Compute the finite edge-path fundamental-group presentation",
        description=(
            "For one bounded finite simplicial complex and a selected base "
            "vertex, collapse the basepoint 1-component along one "
            "lexicographic breadth-first spanning tree, assign a generator to "
            "every non-tree edge, and translate each oriented two-simplex "
            "boundary into one freely reduced relator. Return the exact "
            "spanning tree, generator and oriented-edge word maps, triangle "
            "relators, the finite presentation, and its exact integer "
            "abelianization (relation matrix, Smith rank, invariant factors). "
            "Disconnected input requires the explicit basepoint component."
        ),
        request_type=FundamentalGroupPresentationRequest,
        result_type=FundamentalGroupPresentationResult,
        run=_fundamental_group,
        tags=("topology", "simplicial", "fundamental-group", "exact"),
        discovery_terms=(
            "fundamental group presentation",
            "edge path group",
            "spanning tree generators",
            "triangle relators",
            "abelianization invariants",
        ),
        examples=(
            OperationExample(
                name="triangulated_circle",
                description=(
                    "Three edges around a triangle as a 1-dimensional circle: "
                    "one non-tree generator and no relator."
                ),
                input={
                    "complex": {
                        "vertices": ["0", "1", "2"],
                        "facets": [["0", "1"], ["1", "2"], ["0", "2"]],
                    },
                    "base_vertex": "0",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
