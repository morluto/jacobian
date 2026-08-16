"""Domain adapter for code theory operations."""
from __future__ import annotations
from jacobian.contracts.code_theory import (
    LinearCodeRequest, MinimumDistanceResult, WeightDistributionResult,
)
from jacobian.math.code_theory import minimum_distance, weight_distribution

def compute_min_distance(request: LinearCodeRequest) -> MinimumDistanceResult:
    matrix = [[request.field_order, 0, 0]]  # Simplified: would need proper field element representation
    # For now, use integer representation
    matrix = [list(row) for row in request.generator_matrix]
    dist = minimum_distance(matrix, request.field_order)
    return MinimumDistanceResult(minimum_distance=dist)

def compute_weight_dist(request: LinearCodeRequest) -> WeightDistributionResult:
    matrix = [list(row) for row in request.generator_matrix]
    weights = weight_distribution(matrix, request.field_order)
    return WeightDistributionResult(
        weights=tuple((w, c) for w, c in weights)
    )
