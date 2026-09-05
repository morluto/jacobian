"""Edge-clique partition operation declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationExample,
)
from jacobian.math.graphs.clique_partition._models import (
    EdgeCliquePartitionRequest,
    EdgeCliquePartitionResult,
)
from jacobian.math.graphs.clique_partition.operations import (
    check_edge_clique_partition,
)


def _compute_edge_clique_partition(
    request: EdgeCliquePartitionRequest,
) -> EdgeCliquePartitionResult:
    return check_edge_clique_partition(request.graph, request.parts)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.edge_clique_partition.check",
        title="Check a supplied edge-clique partition",
        description=(
            "Check that supplied vertex subsets are cliques of order at least "
            "two and that every graph edge occurs in exactly one part. "
            "Returns the valid verdict or one concrete failure: a nonedge "
            "inside a part, an uncovered edge, or an edge covered by multiple "
            "identified parts. Certifies an upper bound on clique-partition "
            "number, not minimality."
        ),
        request_type=EdgeCliquePartitionRequest,
        result_type=EdgeCliquePartitionResult,
        run=_compute_edge_clique_partition,
        tags=("graph", "clique", "exact"),
        examples=(
            OperationExample(
                name="diamond_valid_partition",
                description=(
                    "Partition the diamond's edges into triangle abc and edges ad, bd."
                ),
                input={
                    "graph": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [
                            ["a", "b"],
                            ["a", "c"],
                            ["a", "d"],
                            ["b", "c"],
                            ["b", "d"],
                        ],
                    },
                    "parts": [["a", "b", "c"], ["a", "d"], ["b", "d"]],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
