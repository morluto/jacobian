"""Rational subset-sum profile operation declarations."""

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


TOOLS: MathTools = (
    MathTool(
        operation_id="additive.rational_subset_sum.profile.compute",
        title="Compute the rational subset-sum profile",
        description=(
            "Given an ordered indexed sequence of canonical rationals, return "
            "every attainable subset sum and its number of realizing selection "
            "vectors, including the empty subset at zero."
        ),
        request_type=RationalSubsetSumRequest,
        result_type=RationalSubsetSumResult,
        run=compute_rational_subset_sum_profile_op,
        tags=("additive-combinatorics", "exact"),
        examples=(
            OperationExample(
                name="fixture",
                description="Complete profile of (1/2, 1/2, -1).",
                input={
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
