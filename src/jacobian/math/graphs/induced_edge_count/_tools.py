"""Induced-edge-count profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.induced_edge_count._models import (
    InducedEdgeCountProfileRequest,
    InducedEdgeCountProfileResult,
)
from jacobian.math.graphs.induced_edge_count.operations import (
    compute_induced_edge_count_profile,
)


def compute_induced_edge_count_profile_op(
    request: InducedEdgeCountProfileRequest,
) -> InducedEdgeCountProfileResult:
    return compute_induced_edge_count_profile(request.graph, request.cardinality)


def iec_operation[
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
    iec_operation(
        "graph.induced_edge_count_profile.compute",
        "Profile induced-edge counts over fixed-cardinality vertex subsets",
        (
            "For a finite simple graph and a selected cardinality k, return the "
            "complete exact distribution of induced-edge counts over all "
            "k-element vertex subsets, with one canonical witness subset for "
            "each attained count."
        ),
        InducedEdgeCountProfileRequest,
        InducedEdgeCountProfileResult,
        compute_induced_edge_count_profile_op,
        "graph",
        "profile",
        "exact",
        examples=(
            example(
                "path3_k2",
                "P3 at cardinality 2: subsets have edge counts 1,1,0.",
                {
                    "graph": {
                        "vertices": ["0", "1", "2"],
                        "edges": [["0", "1"], ["1", "2"]],
                    },
                    "cardinality": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
