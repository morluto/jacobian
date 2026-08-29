"""Typed declarations for the common-neighbour profile operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.common_neighbor_profile._models import (
    CommonNeighborProfileRequest,
    CommonNeighborProfileResult,
)
from jacobian.math.graphs.common_neighbor_profile.operations import (
    compute_common_neighbor_profile,
)


def _compute(request: CommonNeighborProfileRequest) -> CommonNeighborProfileResult:
    return compute_common_neighbor_profile(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.invariant.common_neighbor_profile.compute",
        title="Compute the common-neighbour profile of a graph",
        description=(
            "For one bounded finite simple graph, return for every unordered "
            "pair of distinct vertices the canonical sorted set of common "
            "neighbours, its cardinality (codegree), the maximum codegree, "
            "the complete cardinality histogram, and whether the graph is "
            "C4-free."
        ),
        request_type=CommonNeighborProfileRequest,
        result_type=CommonNeighborProfileResult,
        run=_compute,
        tags=("graph", "invariant", "exact", "bounded"),
        examples=(
            example(
                "triangle",
                "The common-neighbour profile of a triangle.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
