"""Typed declarations for the weighted monotone endpoint profile operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.weighted_monotone_profiles._models import (
    WeightedMonotoneProfileRequest,
    WeightedMonotoneProfileResult,
)
from jacobian.math.combinatorics.weighted_monotone_profiles.operations import (
    compute_weighted_monotone_profiles,
)


def _compute(request: WeightedMonotoneProfileRequest) -> WeightedMonotoneProfileResult:
    return compute_weighted_monotone_profiles(request.alphabet, request.weights)


TOOLS: MathTools = (
    MathTool(
        operation_id="algebraic_combinatorics.weighted_monotone_endpoint_profile.compute",
        title="Compute exact weighted monotone endpoint profiles",
        description=(
            "For one bounded ordered finite word with nonnegative rational "
            "weights, return the two exact endpoint DP profiles: the maximum "
            "weight of a weakly increasing subsequence ending at each position, "
            "and the analogous weakly decreasing value."
        ),
        request_type=WeightedMonotoneProfileRequest,
        result_type=WeightedMonotoneProfileResult,
        run=_compute,
        tags=("combinatorics", "monotone", "weighted", "exact"),
        examples=(
            example(
                "simple",
                "Weighted monotone profiles for a small alphabet.",
                {
                    "alphabet": [3, 1, 2, 1, 3],
                    "weights": [
                        {"num": "1", "den": "1"},
                        {"num": "2", "den": "1"},
                        {"num": "3", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "4", "den": "1"},
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
