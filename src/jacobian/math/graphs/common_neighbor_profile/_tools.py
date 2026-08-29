"""Common-neighbour profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.common_neighbor_profile._models import (
    CommonNeighborProfileRequest,
    CommonNeighborProfileResult,
)
from jacobian.math.graphs.common_neighbor_profile.operations import (
    compute_common_neighbor_profile,
)


def compute_common_neighbor_profile_op(
    request: CommonNeighborProfileRequest,
) -> CommonNeighborProfileResult:
    return compute_common_neighbor_profile(request.graph)


def cnp_operation[
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
    cnp_operation(
        "graph.invariant.common_neighbor_profile.compute",
        "Compute the common-neighbour profile of a graph",
        (
            "For a bounded finite simple graph G, return for every unordered pair "
            "of distinct vertices the canonical sorted set of common neighbours "
            "(N(u) ∩ N(v)) and its cardinality (codegree)."
        ),
        CommonNeighborProfileRequest,
        CommonNeighborProfileResult,
        compute_common_neighbor_profile_op,
        "graph",
        "invariant",
        "exact",
        examples=(
            example(
                "c4",
                "The 4-cycle C4 has opposite pairs with codegree 2 and adjacent pairs with codegree 0.",
                {
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [["0", "1"], ["1", "2"], ["2", "3"], ["0", "3"]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
