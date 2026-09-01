"""Edge pattern profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.finite_structures.edge_pattern_profile._models import (
    EdgePatternProfileRequest,
    EdgePatternProfileResult,
)
from jacobian.math.combinatorics.finite_structures.edge_pattern_profile.operations import (
    compute_edge_pattern_profile,
)


def compute_edge_pattern_profile_op(
    request: EdgePatternProfileRequest,
) -> EdgePatternProfileResult:
    return compute_edge_pattern_profile(request.hypergraph, request.vertex_colors)


TOOLS: MathTools = (
    MathTool(
        operation_id="hypergraph.vertex_coloring.edge_pattern_profile.compute",
        title="Compute the edge-pattern profile of a vertex-coloured hypergraph",
        description=(
            "Given one finite hypergraph and one total finite-label colouring "
            "of its vertex axis, return the complete source-edge partition by "
            "the canonical equality pattern of its vertex colours, with the "
            "monochromatic and rainbow edge subfamilies explicit."
        ),
        request_type=EdgePatternProfileRequest,
        result_type=EdgePatternProfileResult,
        run=compute_edge_pattern_profile_op,
        tags=("hypergraph", "exact"),
        examples=(
            OperationExample(
                name="three_edge",
                description="A hypergraph with a mixed vertex colouring.",
                input={
                    "hypergraph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [
                            ["e0", ["a", "b", "c"]],
                            ["e1", ["a", "b"]],
                        ],
                    },
                    "vertex_colors": {"a": "red", "b": "red", "c": "blue"},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
