"""Exact lexicographic matroid intersection optimization."""

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    MAX_WEIGHT_DIGITS,
    MatroidWeightedIntersectionOptimizationRequest,
    MatroidWeightedIntersectionOptimizationResult,
    MatroidWeightFunction,
)
from jacobian.math.combinatorics.matroids.intersection import (
    maximum_weight_matroid_intersection,
)
from jacobian.math.combinatorics.matroids.operations import _canonical_weight_function

from ._models import MatroidCardinalityWeightedIntersectionResult


def maximum_cardinality_weighted_matroid_intersection(
    request: MatroidWeightedIntersectionOptimizationRequest,
) -> MatroidCardinalityWeightedIntersectionResult:
    """Maximize common-set cardinality first and original total weight second.

    A cardinality bonus ``2*n*W + 1``, where ``W=max(abs(w_e))``, makes every
    one-element cardinality gain outweigh the largest possible original-weight
    loss. The existing exact weighted-intersection solver then optimizes the
    scalarized objective and supplies its ordinary split witness.
    """
    if type(request) is not MatroidWeightedIntersectionOptimizationRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="matroid.cardinality_weighted_intersection.request",
            message=(
                "request must be a canonical weighted-intersection optimization request"
            ),
        )
    try:
        request = MatroidWeightedIntersectionOptimizationRequest.model_validate(
            request.model_dump(mode="python")
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="matroid.cardinality_weighted_intersection.request",
            message="request must satisfy the weighted-intersection request schema",
        ) from exc
    n = request.first.ground_size
    weights, original_function = _canonical_weight_function(
        request.first, request.weight_function
    )
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

    shifted_request = MatroidWeightedIntersectionOptimizationRequest(
        first=request.first,
        second=request.second,
        weight_function=MatroidWeightFunction(
            ground_axis=original_function.ground_axis,
            values=shifted_values,
        ),
    )
    optimized: MatroidWeightedIntersectionOptimizationResult = (
        maximum_weight_matroid_intersection(shifted_request)
    )
    selected = optimized.common_independent
    return MatroidCardinalityWeightedIntersectionResult.model_construct(
        optimized=optimized,
        cardinality_bonus=bonus,
        cardinality=len(selected),
        total_weight=sum(weights[index] for index in selected),
    )
