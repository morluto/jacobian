"""Homogeneous progression set system operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.discrepancy.homogeneous_progression._models import (
    HomogeneousProgressionRequest,
    HomogeneousProgressionResult,
)
from jacobian.math.combinatorics.discrepancy.homogeneous_progression.operations import (
    construct_homogeneous_progression_set_system,
)


def compute_homogeneous_progression(
    request: HomogeneousProgressionRequest,
) -> HomogeneousProgressionResult:
    return construct_homogeneous_progression_set_system(request.n)


def hp_operation[
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
    hp_operation(
        "discrepancy.homogeneous_progression_set_system.construct",
        "Construct the homogeneous progression set system on [n]",
        (
            "Construct the canonical finite set system whose ground set is [n] "
            "and whose sets are the homogeneous arithmetic progressions "
            "{d, 2d, ..., kd} for every d, k >= 1 with dk <= n, with zero-based "
            "indexing. This is the standard carrier for Erdős discrepancy "
            "experiments."
        ),
        HomogeneousProgressionRequest,
        HomogeneousProgressionResult,
        compute_homogeneous_progression,
        "combinatorics",
        "discrepancy",
        "exact",
        examples=(
            example(
                "n4",
                "The homogeneous progression set system on [4].",
                {"n": 4},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
