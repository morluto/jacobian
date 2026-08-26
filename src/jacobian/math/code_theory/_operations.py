"""Domain-owned code theory operations."""

from __future__ import annotations

from jacobian.math.code_theory._models import (
    CoveringRadiusRequest,
    CoveringRadiusResult,
    LinearCodeRequest,
    MinimumDistanceResult,
    WeightDistributionResult,
)
from jacobian.math.code_theory.operations import (
    covering_radius,
    minimum_distance,
    weight_distribution,
)


def compute_min_distance(request: LinearCodeRequest) -> MinimumDistanceResult:
    dist = minimum_distance(request.generator_matrix, request.field_order)
    return MinimumDistanceResult._from_kernel(
        request=request,
        minimum_distance=dist,
    )


def compute_weight_dist(request: LinearCodeRequest) -> WeightDistributionResult:
    weights = weight_distribution(request.generator_matrix, request.field_order)
    return WeightDistributionResult._from_kernel(
        request=request,
        weights=tuple((w, c) for w, c in weights),
    )


def compute_covering_radius(request: CoveringRadiusRequest) -> CoveringRadiusResult:
    radius = covering_radius(request.generator_matrix, request.field_order)
    return CoveringRadiusResult._from_kernel(
        request=request,
        covering_radius=radius,
    )


def verify_minimum_distance_result(result: MinimumDistanceResult) -> bool:
    """Replay an independently supplied minimum-distance claim."""

    return (
        minimum_distance(result.request.generator_matrix, result.request.field_order)
        == result.minimum_distance
    )


def verify_weight_distribution_result(result: WeightDistributionResult) -> bool:
    """Replay an independently supplied weight-distribution claim."""

    expected = weight_distribution(
        result.request.generator_matrix,
        result.request.field_order,
    )
    return tuple(expected) == result.weights


def verify_covering_radius_result(result: CoveringRadiusResult) -> bool:
    """Replay an independently supplied covering-radius claim."""

    return (
        covering_radius(result.request.generator_matrix, result.request.field_order)
        == result.covering_radius
    )
