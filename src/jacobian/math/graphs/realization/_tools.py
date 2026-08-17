"""Exact graph realization operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.realization._models import (
    DegreeSequenceRequest,
    DegreeSequenceResult,
    GraphRealizationRequest,
    GraphRealizationResult,
    RealizationCheckRequest,
    RealizationCheckResult,
)
from jacobian.math.graphs.realization._operations import (
    compute_degree_sequence,
    compute_graph_realization,
    compute_realization_check,
)


def graph_realization_operation[
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


GRAPH_REALIZATION_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    graph_realization_operation(
        "graph.realization.is_graphical.compute",
        "Determine if a degree sequence is graphical",
        "Determine if a degree sequence is graphical using the Erdos-Gallai "
        "theorem. A sequence is graphical if there exists a simple graph whose "
        "vertex degrees exactly match the sequence.",
        DegreeSequenceRequest,
        DegreeSequenceResult,
        compute_degree_sequence,
        "graph",
        "realization",
        "graphicality",
        "exact",
        examples=(
            example(
                "graphical_path",
                "A degree sequence that realizes a simple path on 4 vertices.",
                {"sequence": {"degrees": [1, 2, 2, 1]}},
            ),
            example(
                "non_graphical_odd_sum",
                "An odd degree sum is never graphical.",
                {"sequence": {"degrees": [3, 3, 3]}},
            ),
        ),
    ),
    graph_realization_operation(
        "graph.realization.construct.compute",
        "Construct a simple graph realizing a degree sequence",
        "Construct a simple undirected graph that realizes a graphical degree "
        "sequence using the Havel-Hakimi algorithm. Returns the edges of the "
        "realized graph; if the sequence is not graphical, no edges are returned.",
        GraphRealizationRequest,
        GraphRealizationResult,
        compute_graph_realization,
        "graph",
        "realization",
        "construction",
        "exact",
        examples=(
            example(
                "realize_path",
                "Construct a simple path on 4 vertices from its degree sequence.",
                {"sequence": {"degrees": [1, 2, 2, 1]}},
            ),
            example(
                "realize_cycle",
                "Construct a 4-cycle from its degree sequence.",
                {"sequence": {"degrees": [2, 2, 2, 2]}},
            ),
        ),
    ),
    graph_realization_operation(
        "graph.realization.check.compute",
        "Verify that a graph realizes a degree sequence",
        "Verify that a given simple undirected graph (vertex_count + edges) "
        "realizes a degree sequence by computing the graph's vertex degrees and "
        "comparing them to the expected sequence.",
        RealizationCheckRequest,
        RealizationCheckResult,
        compute_realization_check,
        "graph",
        "realization",
        "check",
        "exact",
        examples=(
            example(
                "valid_realization",
                "A valid realization of the degree sequence [1, 2, 2, 1].",
                {
                    "sequence": {"degrees": [1, 2, 2, 1]},
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                },
            ),
        ),
    ),
)

TOOLS = GRAPH_REALIZATION_OPERATIONS

__all__ = ["TOOLS"]
