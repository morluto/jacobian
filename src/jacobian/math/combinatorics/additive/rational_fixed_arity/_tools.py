"""Rational fixed-arity sum profile operation declarations."""

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


TOOLS: MathTools = (
    MathTool(
        operation_id="additive.rational_fixed_arity_sum.profile.compute",
        title="Compute the rational fixed-arity sum profile",
        description=(
            "Given an ordered finite sequence of canonical rational values and "
            "an arity h, return every attained rational sum and the number of "
            "strictly increasing source-index h-tuples attaining it."
        ),
        request_type=RationalFixedAritySumRequest,
        result_type=RationalFixedAritySumResult,
        run=compute_rational_fixed_arity_sum_profile_op,
        tags=("additive-combinatorics", "exact"),
        examples=(
            OperationExample(
                name="unit_fractions",
                description="Sums of pairs from (1/2, 1/3, 1/6).",
                input={
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
