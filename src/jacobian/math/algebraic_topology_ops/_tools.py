"""Algebraic topology operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.algebraic_topology_ops._models import (
    EdgePathConcatenateRequest,
    EdgePathConcatenateResult,
    EdgePathWordRequest,
    EdgePathWordResult,
)
from jacobian.math.algebraic_topology_ops._operations import (
    compute_edge_path_concatenate,
    compute_edge_path_word,
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
        "topology.simplicial.edge_path.word.compute",
        "Compute the free group word for an edge path",
        "Compute the free group word representation of an edge path in a "
        "graph, where each edge corresponds to a generator and its inverse.",
        EdgePathWordRequest,
        EdgePathWordResult,
        compute_edge_path_word,
        "topology",
        "edge-path",
        "exact",
        examples=(
            example(
                "triangle_path",
                "Compute the word for path 0->1->2 in a triangle.",
                {
                    "vertex_count": 3,
                    "edges": [[0, 1], [1, 2], [2, 0]],
                    "path": [0, 1, 2],
                },
            ),
        ),
    ),
    _op(
        "topology.simplicial.edge_path.concatenate.compute",
        "Concatenate two edge paths",
        "Concatenate two edge paths in a graph, removing the shared vertex.",
        EdgePathConcatenateRequest,
        EdgePathConcatenateResult,
        compute_edge_path_concatenate,
        "topology",
        "edge-path",
        "exact",
        examples=(
            example(
                "concatenate_paths",
                "Concatenate [0,1] and [1,2] in a 3-vertex graph.",
                {
                    "vertex_count": 3,
                    "path_a": [0, 1],
                    "path_b": [1, 2],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
