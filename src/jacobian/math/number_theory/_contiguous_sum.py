"""Declarations for contiguous-sum representation profiles."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileRequest,
    ContiguousSumProfileResult,
)
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.operations import contiguous_sum_profile


def compute_contiguous_sum_profile(
    request: ContiguousSumProfileRequest,
) -> ContiguousSumProfileResult:
    return contiguous_sum_profile(request.lower_bound, request.upper_bound)


CONTIGUOUS_SUM_OPERATION = number_theory_operation(
    "number_theory.integer_interval.contiguous_sum_profile.compute",
    "Compute contiguous-sum representation profile on a bounded interval",
    "For each n in [L, U], count representations as a sum of consecutive positive integers, or report UNKNOWN if high-magnitude factorization does not complete within its bounded worker envelope.",
    ContiguousSumProfileRequest,
    ContiguousSumProfileResult,
    compute_contiguous_sum_profile,
    "number-theory",
    "arithmetic-function",
    "interval-profile",
    examples=(
        example(
            "contiguous_sum_1_15",
            "Count contiguous-sum representations for n from 1 to 15. "
            "The interval must contain at most 100,000 integers.",
            {"lower_bound": "1", "upper_bound": "15"},
        ),
    ),
)

__all__ = ["CONTIGUOUS_SUM_OPERATION"]
