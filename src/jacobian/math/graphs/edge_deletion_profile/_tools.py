"""Edge deletion profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.edge_deletion_profile._models import (
    EdgeDeletionProfileRequest,
    EdgeDeletionProfileResult,
)
from jacobian.math.graphs.edge_deletion_profile.operations import (
    compute_edge_deletion_profile,
)


def compute_edge_deletion_profile_op(
    request: EdgeDeletionProfileRequest,
) -> EdgeDeletionProfileResult:
    return compute_edge_deletion_profile(request.graph, request.deletion_order)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.coloring.edge_deletion_profile.compute",
        title="Compute the edge-deletion chromatic profile of a graph",
        description=(
            "Given a finite simple graph G and a nonnegative deletion order b, "
            "return the exact chromatic number of G-F for every edge subset "
            "F with |F| <= b, indexed by canonical edge-subset values."
        ),
        request_type=EdgeDeletionProfileRequest,
        result_type=EdgeDeletionProfileResult,
        run=compute_edge_deletion_profile_op,
        tags=("graph", "coloring", "exact"),
        examples=(
            OperationExample(
                name="k3_order1",
                description=(
                    "K3 with deletion order 1: returns the chromatic number "
                    "of G-F for every edge subset F with |F| <= 1. "
                    "deletion_order must not exceed the graph's edge count."
                ),
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                    },
                    "deletion_order": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
