"""Common-neighbour profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.common_neighbor_profile._models import (
    CommonNeighborProfileRequest,
    CommonNeighborProfileResult,
)
from jacobian.math.graphs.common_neighbor_profile.operations import (
    compute_common_neighbor_profile,
)


def compute_common_neighbor_profile_op(
    request: CommonNeighborProfileRequest,
) -> CommonNeighborProfileResult:
    return compute_common_neighbor_profile(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.invariant.common_neighbor_profile.compute",
        title="Compute the common-neighbour profile of a graph",
        description=(
            "For a bounded finite simple graph G, return for every unordered pair "
            "of distinct vertices the canonical sorted set of common neighbours "
            "(N(u) ∩ N(v)) and its cardinality (codegree)."
        ),
        request_type=CommonNeighborProfileRequest,
        result_type=CommonNeighborProfileResult,
        run=compute_common_neighbor_profile_op,
        tags=("graph", "invariant", "exact"),
        examples=(
            OperationExample(
                name="c4",
                description="The 4-cycle C4 has opposite pairs with codegree 2 and adjacent pairs with codegree 0.",
                input={
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [["0", "1"], ["1", "2"], ["2", "3"], ["0", "3"]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
