"""Cycle-length profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.cycle_length_profile._models import (
    CycleLengthProfileRequest,
    CycleLengthProfileResult,
)
from jacobian.math.graphs.cycle_length_profile.operations import (
    compute_cycle_length_profile,
)


def compute_cycle_length_profile_op(
    request: CycleLengthProfileRequest,
) -> CycleLengthProfileResult:
    return compute_cycle_length_profile(request.graph)


def clp_operation[
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
    clp_operation(
        "graph.invariant.cycle_length_profile.compute",
        "Compute the complete cycle-length profile of a graph",
        (
            "Given a bounded finite simple graph, return the complete set of "
            "simple-cycle lengths together with one canonical witness cycle "
            "for each occurring length. Admission requires the first-witness "
            "search to fit 10,000,000 work units and the complete canonical "
            "result to fit the 10 MiB output envelope."
        ),
        CycleLengthProfileRequest,
        CycleLengthProfileResult,
        compute_cycle_length_profile_op,
        "graph",
        "invariant",
        "exact",
        examples=(
            example(
                "c4",
                "C4 has cycle-length spectrum {4}.",
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
