"""Typed declarations for the rational subset-sum profile operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.additive.rational_subset_sum._models import (
    RationalSubsetSumRequest,
    RationalSubsetSumResult,
)
from jacobian.math.combinatorics.additive.rational_subset_sum.operations import (
    compute_rational_subset_sum_profile,
)


def _compute(request: RationalSubsetSumRequest) -> RationalSubsetSumResult:
    return compute_rational_subset_sum_profile(request.values)


TOOLS: MathTools = (
    MathTool(
        operation_id="additive.rational_subset_sum.profile.compute",
        title="Compute complete indexed rational subset-sum profiles",
        description=(
            "Given an ordered indexed sequence of canonical rationals, return "
            "every attainable subset sum and its number of realizing selection "
            "vectors."
        ),
        request_type=RationalSubsetSumRequest,
        result_type=RationalSubsetSumResult,
        run=_compute,
        tags=("additive", "subset", "sum", "exact"),
        examples=(
            example(
                "simple",
                "Subset sums of {1/2, 1/3}.",
                {
                    "values": [
                        {"num": "1", "den": "2"},
                        {"num": "1", "den": "3"},
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
