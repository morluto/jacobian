"""Operation declaration for edge-deletion diameter profile."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.edge_deletion_diameter_profile._models import (
    EdgeDeletionDiameterProfileRequest,
    EdgeDeletionDiameterProfileResult,
)
from jacobian.math.graphs.edge_deletion_diameter_profile.operations import (
    edge_deletion_diameter_profile,
)


def _run(
    request: EdgeDeletionDiameterProfileRequest,
) -> EdgeDeletionDiameterProfileResult:
    return edge_deletion_diameter_profile(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.edge_deletion_diameter_profile.compute",
        title="Compute the edge-deletion diameter profile of a graph",
        description=(
            "For a nonempty connected simple graph G, return its diameter and the "
            "exact diameter of G-e for every source edge e, or DISCONNECTED when "
            f"deletion disconnects G. The operation admits at most {64} vertices and "
            f"{256} edges and performs at most O(m·(n+m)) BFS work, reusing the "
            "exact NetworkX diameter kernel."
        ),
        request_type=EdgeDeletionDiameterProfileRequest,
        result_type=EdgeDeletionDiameterProfileResult,
        run=_run,
        tags=("graph", "diameter", "edge-deletion", "profile", "exact"),
        examples=(
            OperationExample(
                name="path_three_vertices",
                description=(
                    "P3 has diameter 2; deleting either edge disconnects it. The graph "
                    "must be simple, connected, and nonempty."
                ),
                input={
                    "graph": {
                        "vertices": ["0", "1", "2"],
                        "edges": [["0", "1"], ["1", "2"]],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
