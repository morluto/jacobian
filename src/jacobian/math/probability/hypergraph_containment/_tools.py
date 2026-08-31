"""Hypergraph vertex containment operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def hvc_action[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: MathTools = (
    hvc_action(
        "probability.hypergraph_vertex_containment.compute",
        "Compute the vertex-containment probability profile of a hypergraph",
        (
            "For one finite hypergraph and one uniform exact vertex-retention "
            "probability, return the complete cardinality profile of vertex "
            "subsets that contain a declared hyperedge and its exact "
            "product-measure probability."
        ),
        HypergraphVertexContainmentRequest,
        HypergraphVertexContainmentResult,
        compute_hvc_op,
        "probability",
        "exact",
        examples=(
            example(
                "single_edge",
                "A single-edge hypergraph on 2 vertices with p=1/2.",
                {
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
