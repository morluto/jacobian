"""Hypergraph vertex containment operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.probability.hypergraph_containment._models import (
    HypergraphVertexContainmentRequest,
    HypergraphVertexContainmentResult,
)
from jacobian.math.probability.hypergraph_containment.operations import (
    compute_hypergraph_vertex_containment,
)


def compute_hvc_op(
    request: HypergraphVertexContainmentRequest,
) -> HypergraphVertexContainmentResult:
    return compute_hypergraph_vertex_containment(
        request.hypergraph, request.retention_probability
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="probability.hypergraph_vertex_containment.compute",
        title="Compute the vertex-containment probability profile of a hypergraph",
        description=(
            "For one finite hypergraph and one uniform exact vertex-retention "
            "probability, return the complete cardinality profile of vertex "
            "subsets that contain a declared hyperedge and its exact "
            "product-measure probability."
        ),
        request_type=HypergraphVertexContainmentRequest,
        result_type=HypergraphVertexContainmentResult,
        run=compute_hvc_op,
        tags=("probability", "exact"),
        examples=(
            OperationExample(
                name="single_edge",
                description="A single-edge hypergraph on 2 vertices with p=1/2.",
                input={
                    "hypergraph": {
                        "vertices": ["a", "b"],
                        "edges": [["e0", ["a", "b"]]],
                    },
                    "retention_probability": {"num": "1", "den": "2"},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
