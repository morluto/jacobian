"""Graph and delta-matroid interoperability operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.delta_matroids._models import (
    LoopedGraphDeltaMatroidRequest,
    LoopedGraphDeltaMatroidResult,
)
from jacobian.math.graphs.delta_matroids.operations import (
    looped_adjacency_delta_matroid,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="graph.looped_adjacency_delta_matroid.compute",
        title="Construct a delta-matroid from a looped graph",
        description=(
            "Form the symmetric GF(2) adjacency matrix on the graph's ordered "
            "vertex axis, with loops as diagonal entries, then return the "
            "complete family of vertex subsets inducing nonsingular principal "
            "submatrices. The exact binary principal-minor work bound limits "
            "this operation to at most 8 vertices."
        ),
        request_type=LoopedGraphDeltaMatroidRequest,
        result_type=LoopedGraphDeltaMatroidResult,
        run=lambda request: looped_adjacency_delta_matroid(request.graph),
        tags=("graph", "delta-matroid", "binary", "exact"),
        examples=(
            OperationExample(
                name="one_looped_edge",
                description="Build the principal-minor delta-matroid of a looped edge.",
                input={
                    "graph": {
                        "vertices": ["a", "b"],
                        "edges": [["a", "b"]],
                        "loops": ["a"],
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
