"""Exact graph coloring operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.coloring._models import (
    KColorabilityRequest,
    KColorabilityResult,
    MaximalIndependentSetRequest,
    MaximalIndependentSetResult,
)
from jacobian.math.graphs.coloring._operations import (
    compute_k_colorability,
    compute_maximal_independent_set_decision,
)


def graph_coloring_operation[
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


GRAPH_COLORING_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    graph_coloring_operation(
        "graph.coloring.k_colorability.decide",
        "Decide k-colorability of a graph",
        "Decide whether a simple undirected graph admits a proper k-coloring and return a coloring if one exists, using a Z3 SAT encoding.",
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
        "graph.independent_set.maximal.decide",
        "Decide whether a candidate set is a maximal independent set",
        "Decide maximal independence in a bounded simple graph and return a blocking edge or addable vertex when the candidate fails.",
        MaximalIndependentSetRequest,
        MaximalIndependentSetResult,
        compute_maximal_independent_set_decision,
        "graph",
        "independent-set",
        "maximal",
        "exact",
        examples=(
            example(
                "path_maximal_independent_set",
                "Decide whether {0, 2} is a maximal independent set of P4.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                    "candidate_set": [0, 2],
                },
            ),
        ),
    ),
)

TOOLS = GRAPH_COLORING_OPERATIONS

__all__ = ["TOOLS"]
