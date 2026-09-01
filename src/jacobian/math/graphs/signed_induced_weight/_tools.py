"""Signed induced-weight extrema operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.signed_induced_weight._models import (
    SignedInducedWeightRequest,
    SignedInducedWeightResult,
)
from jacobian.math.graphs.signed_induced_weight.operations import (
    signed_induced_weight_extrema,
)


def compute_signed_induced_weight_extrema(
    request: SignedInducedWeightRequest,
) -> SignedInducedWeightResult:
    return signed_induced_weight_extrema(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.signed_induced_weight.extrema.compute",
        title="Compute exact signed induced-edge weight extrema over all vertex subsets",
        description=(
            "For a finite undirected graph with exact rational edge weights, "
            "return the minimum and maximum of the signed induced-edge total "
            "sum(weight(u,v) : {u,v} is an edge and both endpoints are selected) "
            "over all vertex subsets, with one deterministic witness subset for "
            "each extremum. Empty and singleton subsets have weight zero. The "
            "current exhaustive envelope admits at most 20 vertices and applies "
            "digit-sensitive arithmetic work and exact-result height bounds."
        ),
        request_type=SignedInducedWeightRequest,
        result_type=SignedInducedWeightResult,
        run=compute_signed_induced_weight_extrema,
        tags=("graph", "optimization", "exact"),
        examples=(
            OperationExample(
                name="triangle_signed",
                description=(
                    "Compute the minimum and maximum signed induced weights for "
                    "this triangle. The graph must be simple and its edge weights "
                    "must be canonical rational values."
                ),
                input={
                    "graph": {
                        "vertices": ["0", "1", "2"],
                        "edges": [
                            {
                                "endpoints": ["0", "1"],
                                "weight": {"num": "2", "den": "1"},
                            },
                            {
                                "endpoints": ["0", "2"],
                                "weight": {"num": "-1", "den": "1"},
                            },
                            {
                                "endpoints": ["1", "2"],
                                "weight": {"num": "-1", "den": "1"},
                            },
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
