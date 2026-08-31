"""Edge pattern profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def epp_operation[RequestT: StrictModel, ResultT: StrictModel](
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
    epp_operation(
        "hypergraph.vertex_coloring.edge_pattern_profile.compute",
        "Compute the edge-pattern profile of a vertex-coloured hypergraph",
        (
            "Given one finite hypergraph and one total finite-label colouring "
            "of its vertex axis, return the complete source-edge partition by "
            "the canonical equality pattern of its vertex colours, with the "
            "monochromatic and rainbow edge subfamilies explicit."
        ),
        EdgePatternProfileRequest,
        EdgePatternProfileResult,
        compute_edge_pattern_profile_op,
        "hypergraph",
        "exact",
        examples=(
            example(
                "three_edge",
                "A hypergraph with a mixed vertex colouring.",
                {
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
