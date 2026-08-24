"""Domain-owned code theory operations."""

from __future__ import annotations

from jacobian.math.code_theory import (
    covering_radius,
    minimum_distance,
    weight_distribution,
)
from jacobian.math.code_theory._models import (
    CoveringRadiusRequest,
    CoveringRadiusResult,
    LinearCodeRequest,
    MinimumDistanceResult,
    WeightDistributionResult,
)


def compute_min_distance(request: LinearCodeRequest) -> MinimumDistanceResult:
    dist = minimum_distance(request.generator_matrix, request.field_order)
    return MinimumDistanceResult(request=request, minimum_distance=dist)


def compute_weight_dist(request: LinearCodeRequest) -> WeightDistributionResult:
    weights = weight_distribution(request.generator_matrix, request.field_order)
    return WeightDistributionResult(
        request=request, weights=tuple((w, c) for w, c in weights)
    )


def compute_covering_radius(request: CoveringRadiusRequest) -> CoveringRadiusResult:
    radius = covering_radius(request.generator_matrix, request.field_order)
    return CoveringRadiusResult(request=request, covering_radius=radius)
