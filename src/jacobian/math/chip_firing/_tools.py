"""Chip-firing operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.chip_firing._models import (
    FiringRequest,
    FiringResult,
    LaplacianRequest,
    LaplacianResult,
)
from jacobian.math.chip_firing._operations import compute_firing, compute_laplacian


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
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_GRAPH = {"vertices": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "graph.chip_firing.laplacian.compute",
        "Compute the graph Laplacian",
        "Compute the exact graph Laplacian matrix L = D - A where D is "
        "the degree matrix and A is the adjacency matrix, with vertex "
        "labels and degree vector.",
        LaplacianRequest,
        LaplacianResult,
        compute_laplacian,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "path_graph_3",
                "Compute the Laplacian of a path graph on 3 vertices; "
                "the graph must be a finite undirected simple graph.",
                {"graph": _GRAPH},
            ),
        ),
    ),
    _op(
        "graph.chip_firing.fire_vertex.compute",
        "Fire a vertex in a chip configuration",
        "Fire a vertex v in a chip configuration: v loses degree(v) "
        "chips and each neighbor gains one chip per edge. Returns "
        "the transformed divisor.",
        FiringRequest,
        FiringResult,
        compute_firing,
        "graph-theory",
        "chip-firing",
        "exact",
        examples=(
            example(
                "fire_vertex_b",
                "Fire vertex b in a path graph; "
                "the divisor length must match the vertex count.",
                {"graph": _GRAPH, "divisor": [3, 0, 1], "firing_vertex": "b"},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
