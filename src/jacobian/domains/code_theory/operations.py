"""Domain adapter for code theory operations."""

from __future__ import annotations

from jacobian.contracts.code_theory import (
    LinearCodeRequest,
    MinimumDistanceResult,
    WeightDistributionResult,
)
from jacobian.math.code_theory import minimum_distance, weight_distribution


def compute_min_distance(request: LinearCodeRequest) -> MinimumDistanceResult:
    dist = minimum_distance(request.generator_matrix, request.field_order)  # type: ignore[no-untyped-call]
    return MinimumDistanceResult(minimum_distance=dist)


def compute_weight_dist(request: LinearCodeRequest) -> WeightDistributionResult:
    weights = weight_distribution(request.generator_matrix, request.field_order)  # type: ignore[no-untyped-call]
    return WeightDistributionResult(weights=tuple((w, c) for w, c in weights))
