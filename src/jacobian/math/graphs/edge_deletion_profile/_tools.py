"""Typed declarations for the edge-deletion chromatic profile operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.edge_deletion_profile._models import (
    EdgeDeletionProfileRequest,
    EdgeDeletionProfileResult,
)
from jacobian.math.graphs.edge_deletion_profile.operations import (
    compute_edge_deletion_profile,
)


def _compute(request: EdgeDeletionProfileRequest) -> EdgeDeletionProfileResult:
    return compute_edge_deletion_profile(request.graph, request.deletion_order)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.coloring.edge_deletion_profile.compute",
        title="Compute bounded edge-deletion chromatic profiles",
        description=(
            "Given a finite simple graph G and a nonnegative deletion order b, "
            "return the exact chromatic number of G - F for every edge subset "
            "F ⊆ E(G) with |F| <= b."
        ),
        request_type=EdgeDeletionProfileRequest,
        result_type=EdgeDeletionProfileResult,
        run=_compute,
        tags=("graph", "coloring", "chromatic", "exact"),
        examples=(
            example(
                "triangle_d0",
                "Chromatic profile of a triangle with zero deletions.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                    "deletion_order": 0,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
