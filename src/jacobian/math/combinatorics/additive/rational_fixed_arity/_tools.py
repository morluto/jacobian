"""Rational fixed-arity sum profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.additive.rational_fixed_arity._models import (
    RationalFixedAritySumRequest,
    RationalFixedAritySumResult,
)
from jacobian.math.combinatorics.additive.rational_fixed_arity.operations import (
    compute_rational_fixed_arity_sum_profile,
)


def compute_rational_fixed_arity_sum_profile_op(
    request: RationalFixedAritySumRequest,
) -> RationalFixedAritySumResult:
    return compute_rational_fixed_arity_sum_profile(request.values, request.arity)


def rfa_operation[
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
    rfa_operation(
        "additive.rational_fixed_arity_sum.profile.compute",
        "Compute the rational fixed-arity sum profile",
        (
            "Given an ordered finite sequence of canonical rational values and "
            "an arity h, return every attained rational sum and the number of "
            "strictly increasing source-index h-tuples attaining it."
        ),
        RationalFixedAritySumRequest,
        RationalFixedAritySumResult,
        compute_rational_fixed_arity_sum_profile_op,
        "additive-combinatorics",
        "exact",
        examples=(
            example(
                "unit_fractions",
                "Sums of pairs from (1/2, 1/3, 1/6).",
                {
                    "values": [
                        {"num": "1", "den": "2"},
                        {"num": "1", "den": "3"},
                        {"num": "1", "den": "6"},
                    ],
                    "arity": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
