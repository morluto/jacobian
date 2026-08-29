"""Path decomposition operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.path_decomposition._models import (
    PathDecompositionRequest,
    PathDecompositionResult,
)
from jacobian.math.graphs.path_decomposition.operations import (
    compute_minimum_path_decomposition,
)


def compute_minimum_path_decomposition_op(
    request: PathDecompositionRequest,
) -> PathDecompositionResult:
    return compute_minimum_path_decomposition(request.graph)


def pd_operation[
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


TOOLS: MathTools = (
    pd_operation(
        "graph.path_decomposition.minimum.compute",
        "Compute the exact minimum path decomposition of a graph",
        (
            "Given a bounded finite simple graph, return its exact path number "
            "(minimum number of edge-disjoint simple paths covering all edges) "
            "together with a realizing partition."
        ),
        PathDecompositionRequest,
        PathDecompositionResult,
        compute_minimum_path_decomposition_op,
        "graph",
        "optimization",
        "exact",
        examples=(
            example(
                "p3",
                "P3 has path number 1 (one path covers all edges).",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
