"""Edge deletion profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.edge_deletion_profile._models import (
    EdgeDeletionProfileRequest,
    EdgeDeletionProfileResult,
)
from jacobian.math.graphs.edge_deletion_profile.operations import (
    compute_edge_deletion_profile,
)


def compute_edge_deletion_profile_op(
    request: EdgeDeletionProfileRequest,
) -> EdgeDeletionProfileResult:
    return compute_edge_deletion_profile(request.graph, request.deletion_order)


def edp_action[RequestT: StrictModel, ResultT: StrictModel](
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
    edp_action(
        "graph.coloring.edge_deletion_profile.compute",
        "Compute the edge-deletion chromatic profile of a graph",
        (
            "Given a finite simple graph G and a nonnegative deletion order b, "
            "return the exact chromatic number of G-F for every edge subset "
            "F with |F| <= b, indexed by canonical edge-subset values."
        ),
        EdgeDeletionProfileRequest,
        EdgeDeletionProfileResult,
        compute_edge_deletion_profile_op,
        "graph",
        "coloring",
        "exact",
        examples=(
            example(
                "k3_order1",
                (
                    "K3 with deletion order 1: returns the chromatic number "
                    "of G-F for every edge subset F with |F| <= 1. "
                    "deletion_order must not exceed the graph's edge count."
                ),
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                    },
                    "deletion_order": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
