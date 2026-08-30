"""Rational subset-sum profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.additive.rational_subset_sum._models import (
    RationalSubsetSumRequest,
    RationalSubsetSumResult,
)
from jacobian.math.combinatorics.additive.rational_subset_sum.operations import (
    compute_rational_subset_sum_profile,
)


def compute_rational_subset_sum_profile_op(
    request: RationalSubsetSumRequest,
) -> RationalSubsetSumResult:
    return compute_rational_subset_sum_profile(request.values)


def rss_operation[
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
    rss_operation(
        "additive.rational_subset_sum.profile.compute",
        "Compute the rational subset-sum profile",
        (
            "Given an ordered indexed sequence of canonical rationals, return "
            "every attainable subset sum and its number of realizing selection "
            "vectors, including the empty subset at zero."
        ),
        RationalSubsetSumRequest,
        RationalSubsetSumResult,
        compute_rational_subset_sum_profile_op,
        "additive-combinatorics",
        "exact",
        examples=(
            example(
                "fixture",
                "Complete profile of (1/2, 1/2, -1).",
                {
                    "values": [
                        {"num": "1", "den": "2"},
                        {"num": "1", "den": "2"},
                        {"num": "-1", "den": "1"},
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
