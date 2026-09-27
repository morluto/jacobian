"""Lexicographic maximum-cardinality then maximum-weight intersection."""

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.matroids._models import (
    MAX_WEIGHT_DIGITS,
    LinearMatroid,
    MatroidWeightedIntersectionOptimizationResult,
    MatroidWeightFunction,
)
from jacobian.math.combinatorics.matroids.intersection import (
    _admit_pair,
    _maximum_weight_matroid_intersection_from_values,
)
from jacobian.math.combinatorics.matroids.operations import _canonical_weight_function

from ._models import MatroidCardinalityWeightedIntersectionResult


def maximum_cardinality_weighted_matroid_intersection(
    first: LinearMatroid,
    second: LinearMatroid,
    weight_function: MatroidWeightFunction,
) -> MatroidCardinalityWeightedIntersectionResult:
    """Maximize common-set cardinality first, then original total weight.

    A cardinality bonus ``2*n*W + 1``, where ``W=max(abs(w_e))``, makes every
    one-element cardinality gain outweigh the largest possible original-weight
    loss. The exact weighted-intersection kernel then optimizes the scalarized
    objective and supplies its ordinary split witness.
    """
    first, second = _admit_pair(first, second)
    weights, original_function = _canonical_weight_function(first, weight_function)
    n = first.ground_size
    maximum_absolute_weight = max((abs(value) for value in weights), default=0)
    bonus = 2 * n * maximum_absolute_weight + 1
    shifted_values = tuple(value + bonus for value in weights)
    if any(abs(value) >= 10**MAX_WEIGHT_DIGITS for value in shifted_values):
        raise OperationResourceAdmissionError(
            location=("weight_function", "values"),
            code="matroid.cardinality_weighted_intersection.shift_bound",
            message=(
                "the exact cardinality shift exceeds the weighted-intersection "
                f"{MAX_WEIGHT_DIGITS}-digit objective envelope"
            ),
        )

    optimized: MatroidWeightedIntersectionOptimizationResult = (
        _maximum_weight_matroid_intersection_from_values(
            first,
            second,
            MatroidWeightFunction(
                ground_axis=original_function.ground_axis,
                values=shifted_values,
            ),
        )
    )
    selected = optimized.common_independent
    return MatroidCardinalityWeightedIntersectionResult.model_construct(
        optimized=optimized,
        cardinality_bonus=bonus,
        cardinality=len(selected),
        total_weight=sum(weights[index] for index in selected),
    )
