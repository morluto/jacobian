"""Domain-owned code theory operations."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.general._models import (
    EXACT_ENUMERATION_PASSES,
    MAX_COVERING_RADIUS_STATES_PER_PASS,
    MAX_COVERING_RADIUS_TRANSITIONS,
    MAX_EXACT_CODEWORD_EVALUATIONS,
    SYNDROME_BFS_PASSES,
    CoveringRadiusRequest,
    CoveringRadiusResult,
    LinearCodeRequest,
    MinimumDistanceResult,
    WeightDistributionResult,
    _matrix_rank_mod_prime,
    _validate_prime_field_matrix,
)
from jacobian.math.combinatorics.codes.general.operations import (
    covering_radius,
    minimum_distance,
    weight_distribution,
)


def compute_min_distance(request: LinearCodeRequest) -> MinimumDistanceResult:
    if (
        EXACT_ENUMERATION_PASSES * request.field_order ** len(request.generator_matrix)
        > MAX_EXACT_CODEWORD_EVALUATIONS
    ):
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.enumeration_work_exceeded",
            message="generator matrix exceeds the exact enumeration bound",
        )
    dist = minimum_distance(request.generator_matrix, request.field_order)
    return MinimumDistanceResult._from_kernel(
        request=request,
        minimum_distance=dist,
    )


def compute_weight_dist(request: LinearCodeRequest) -> WeightDistributionResult:
    if (
        EXACT_ENUMERATION_PASSES * request.field_order ** len(request.generator_matrix)
        > MAX_EXACT_CODEWORD_EVALUATIONS
    ):
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.enumeration_work_exceeded",
            message="generator matrix exceeds the exact enumeration bound",
        )
    weights = weight_distribution(request.generator_matrix, request.field_order)
    return WeightDistributionResult._from_kernel(
        request=request,
        weights=tuple((w, c) for w, c in weights),
    )


def compute_covering_radius(request: CoveringRadiusRequest) -> CoveringRadiusResult:
    width = _validate_prime_field_matrix(request.field_order, request.generator_matrix)
    rank = _matrix_rank_mod_prime(request.generator_matrix, request.field_order)
    state_count = request.field_order ** (width - rank)
    if state_count > MAX_COVERING_RADIUS_STATES_PER_PASS:
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.syndrome_state_bound_exceeded",
            message="syndrome space exceeds the exact state bound",
        )
    move_count_bound = min(width * (request.field_order - 1), max(state_count - 1, 0))
    if (
        SYNDROME_BFS_PASSES * state_count * move_count_bound
        > MAX_COVERING_RADIUS_TRANSITIONS
    ):
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.syndrome_transition_bound_exceeded",
            message="syndrome graph exceeds the exact transition bound",
        )
    radius = covering_radius(request.generator_matrix, request.field_order)
    return CoveringRadiusResult._from_kernel(
        request=request,
        covering_radius=radius,
    )
