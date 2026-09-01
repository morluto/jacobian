"""Declarations for contiguous-sum representation profiles."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileRequest,
    ContiguousSumProfileResult,
)
from jacobian.math.number_theory.operations import contiguous_sum_profile


def compute_contiguous_sum_profile(
    request: ContiguousSumProfileRequest,
) -> ContiguousSumProfileResult:
    return contiguous_sum_profile(request.lower_bound, request.upper_bound)


CONTIGUOUS_SUM_OPERATION = MathTool(
    operation_id="number_theory.integer_interval.contiguous_sum_profile.compute",
    title="Compute contiguous-sum representation profile on a bounded interval",
    description="For each n in [L, U], count representations as a sum of consecutive positive integers, or report UNKNOWN if high-magnitude factorization does not complete within its bounded worker envelope.",
    request_type=ContiguousSumProfileRequest,
    result_type=ContiguousSumProfileResult,
    run=compute_contiguous_sum_profile,
    tags=("number-theory", "arithmetic-function", "interval-profile"),
    examples=(
        OperationExample(
            name="contiguous_sum_1_15",
            description="Count contiguous-sum representations for n from 1 to 15. "
            "The interval must contain at most 100,000 integers.",
            input={"lower_bound": "1", "upper_bound": "15"},
        ),
    ),
)

__all__ = ["CONTIGUOUS_SUM_OPERATION"]
