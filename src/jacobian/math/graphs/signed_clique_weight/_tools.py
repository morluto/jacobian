"""Signed clique-weight operation declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationExample,
)
from jacobian.math.graphs.signed_clique_weight._models import (
    SignedCliqueWeightRequest,
    SignedCliqueWeightResult,
)
from jacobian.math.graphs.signed_clique_weight.operations import (
    signed_clique_weight_maximum,
)


def _compute_signed_clique_weight_maximum(
    request: SignedCliqueWeightRequest,
) -> SignedCliqueWeightResult:
    return signed_clique_weight_maximum(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.signed_clique_weight.maximum.compute",
        title="Maximize signed edge weight over nontrivial cliques",
        description=(
            "Compute the maximum induced signed edge-weight sum over cliques "
            "of order at least two, returning a maximizing clique bound to "
            "the graph and weights. Graphs without edges report a missing "
            "optimum explicitly; all-negative inputs may have a negative "
            "optimum. Maximal-clique-only tests are unsound for signed "
            "weights."
        ),
        request_type=SignedCliqueWeightRequest,
        result_type=SignedCliqueWeightResult,
        run=_compute_signed_clique_weight_maximum,
        tags=("graph", "clique", "exact"),
        examples=(
            OperationExample(
                name="signed_triangle",
                description="K3 with one weight-2 edge and two weight -2 edges.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [
                            {
                                "endpoints": ["a", "b"],
                                "weight": {"num": "2", "den": "1"},
                            },
                            {
                                "endpoints": ["a", "c"],
                                "weight": {"num": "-2", "den": "1"},
                            },
                            {
                                "endpoints": ["b", "c"],
                                "weight": {"num": "-2", "den": "1"},
                            },
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
