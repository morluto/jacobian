"""Induced-edge-count profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.induced_edge_count._models import (
    InducedEdgeCountProfileRequest,
    InducedEdgeCountProfileResult,
)
from jacobian.math.graphs.induced_edge_count.operations import (
    compute_induced_edge_count_profile,
)


def compute_induced_edge_count_profile_op(
    request: InducedEdgeCountProfileRequest,
) -> InducedEdgeCountProfileResult:
    return compute_induced_edge_count_profile(request.graph, request.cardinality)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.induced_edge_count_profile.compute",
        title="Profile induced-edge counts over fixed-cardinality vertex subsets",
        description=(
            "For a finite simple graph and a selected cardinality k, return the "
            "complete exact distribution of induced-edge counts over all "
            "k-element vertex subsets, with one canonical witness subset for "
            "each attained count."
        ),
        request_type=InducedEdgeCountProfileRequest,
        result_type=InducedEdgeCountProfileResult,
        run=compute_induced_edge_count_profile_op,
        tags=("graph", "profile", "exact"),
        examples=(
            OperationExample(
                name="path3_k2",
                description="P3 at cardinality 2: subsets have edge counts 1,1,0.",
                input={
                    "graph": {
                        "vertices": ["0", "1", "2"],
                        "edges": [["0", "1"], ["1", "2"]],
                    },
                    "cardinality": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
