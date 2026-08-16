"""Exact graph coloring operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.graph_coloring_ops import (
    KColorabilityRequest,
    KColorabilityResult,
    MaximumIndependentSetRequest,
    MaximumIndependentSetResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.graph_coloring_ops.operations import (
    compute_k_colorability,
    compute_maximum_independent_set,
)
from jacobian.math_tools import MathTool


def graph_coloring_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
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


GRAPH_COLORING_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    graph_coloring_operation(
        "graph.coloring.k_colorability.decide",
        "Decide k-colorability of a graph",
        "Decide whether a simple undirected graph admits a proper k-coloring and return a coloring if one exists, using NetworkX greedy coloring.",
        KColorabilityRequest,
        KColorabilityResult,
        compute_k_colorability,
        "graph",
        "coloring",
        "k-colorability",
        "exact",
        examples=(
            example(
                "triangle_3_colorable",
                "Decide 3-colorability of a triangle (K3).",
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2], [2, 0]],
                    },
                    "colors": 3,
                },
            ),
        ),
    ),
    graph_coloring_operation(
        "graph.independent_set.maximum.compute",
        "Compute the maximum independent set of a graph",
        "Compute the maximum independent set of a simple undirected graph using NetworkX approximation.",
        MaximumIndependentSetRequest,
        MaximumIndependentSetResult,
        compute_maximum_independent_set,
        "graph",
        "independent-set",
        "exact",
        examples=(
            example(
                "path_max_is",
                "Compute the maximum independent set of a path graph.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                },
            ),
        ),
    ),
)
