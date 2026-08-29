"""Typed declarations for the signed induced-subgraph weight operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.optimization.signed_induced_weight._models import (
    SignedInducedWeightRequest,
    SignedInducedWeightResult,
)
from jacobian.math.graphs.optimization.signed_induced_weight.operations import (
    compute_signed_induced_weight_extrema,
)


def _compute(request: SignedInducedWeightRequest) -> SignedInducedWeightResult:
    return compute_signed_induced_weight_extrema(request.graph, request.edge_weights)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.signed_induced_weight.extrema.compute",
        title="Compute exact signed induced-subgraph weight extrema",
        description=(
            "For a finite undirected graph with exact signed/rational edge "
            "weights, return the min and max induced-edge total over all "
            "selected vertex subsets, with one deterministic witness for each."
        ),
        request_type=SignedInducedWeightRequest,
        result_type=SignedInducedWeightResult,
        run=_compute,
        tags=("graph", "optimization", "exact"),
        examples=(
            example(
                "simple",
                "Signed induced weight extrema of a 3-vertex path.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"]],
                    },
                    "edge_weights": [
                        ["a", "b", {"num": "1", "den": "1"}],
                        ["b", "c", {"num": "-1", "den": "1"}],
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
