"""Path decomposition operation declarations."""

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


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.path_decomposition.minimum.compute",
        title="Compute the exact minimum path decomposition of a graph",
        description=(
            "Given a bounded finite simple graph, return its exact path number "
            "(minimum number of edge-disjoint simple paths covering all edges) "
            "together with a realizing partition."
        ),
        request_type=PathDecompositionRequest,
        result_type=PathDecompositionResult,
        run=compute_minimum_path_decomposition_op,
        tags=("graph", "optimization", "exact"),
        examples=(
            OperationExample(
                name="p3",
                description="P3 has path number 1 (one path covers all edges).",
                input={
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
