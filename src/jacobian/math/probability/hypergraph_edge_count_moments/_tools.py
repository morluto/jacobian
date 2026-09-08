from jacobian.catalog.models import MathTool, MathTools, OperationExample

from ._models import HypergraphEdgeCountMomentsRequest, HypergraphEdgeCountMomentsResult
from .operations import compute_hypergraph_edge_count_moments


def _run(
    request: HypergraphEdgeCountMomentsRequest,
) -> HypergraphEdgeCountMomentsResult:
    return compute_hypergraph_edge_count_moments(
        request.hypergraph, request.retention_probability
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="probability.hypergraph_edge_count_moments.compute",
        title="Compute exact hypergraph edge-count moments",
        description="Return exact first and second moments, variance, and compact edge-overlap covariance profile under independent vertex retention.",
        request_type=HypergraphEdgeCountMomentsRequest,
        result_type=HypergraphEdgeCountMomentsResult,
        run=_run,
        tags=("probability", "combinatorics", "exact"),
        examples=(
            OperationExample(
                name="two_intersecting_edges",
                description="Two edges sharing one vertex at retention probability one half.",
                input={
                    "hypergraph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["e1", ["a", "b"]], ["e2", ["b", "c"]]],
                    },
                    "retention_probability": {"num": "1", "den": "2"},
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
