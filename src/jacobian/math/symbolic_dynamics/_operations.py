"""Wire adapters for public symbolic dynamics operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.symbolic_dynamics._models import (
    ArtinMazurZetaRequest,
    ArtinMazurZetaResult,
    BlockLanguageRequest,
    BlockLanguageResult,
    FiniteTypeShiftRequest,
    FiniteTypeShiftResult,
    HigherBlockRequest,
    HigherBlockResult,
    PeriodicPointProfileRequest,
    PeriodicPointProfileResult,
    _computed_artin_mazur_zeta_result,
)
from jacobian.math.symbolic_dynamics.operations import (
    artin_mazur_zeta,
    block_language,
    finite_type_presentation,
    higher_block_presentation,
    normalize_forbidden_blocks,
    periodic_point_profile,
)


def compute_artin_mazur_zeta(
    request: ArtinMazurZetaRequest,
) -> ArtinMazurZetaResult:
    determinant, zeta, coefficients = artin_mazur_zeta(
        request.shift, request.replay_period
    )
    return _computed_artin_mazur_zeta_result(
        request,
        determinant,
        zeta,
        coefficients,
    )


def construct_finite_type_shift(
    request: FiniteTypeShiftRequest,
) -> FiniteTypeShiftResult:
    return FiniteTypeShiftResult(
        **request.model_dump(),
        presentation=finite_type_presentation(request.shift),
        normalized_forbidden_blocks=normalize_forbidden_blocks(request.shift),
    )


def compute_block_language(request: BlockLanguageRequest) -> BlockLanguageResult:
    allowed = block_language(request.shift, request.block_length)
    return BlockLanguageResult(
        **request.model_dump(), allowed_blocks=allowed, count=len(allowed)
    )


def compute_periodic_point_profile(
    request: PeriodicPointProfileRequest,
) -> PeriodicPointProfileResult:
    fixed, exact, orbits = periodic_point_profile(request.shift, request.max_period)
    return PeriodicPointProfileResult(
        **request.model_dump(),
        periods=tuple(range(1, request.max_period + 1)),
        fixed_point_counts=tuple(format_canonical_integer(value) for value in fixed),
        least_period_point_counts=tuple(
            format_canonical_integer(value) for value in exact
        ),
        primitive_orbit_counts=tuple(
            format_canonical_integer(value) for value in orbits
        ),
        complete_through_period=request.max_period,
    )


def compute_higher_block(request: HigherBlockRequest) -> HigherBlockResult:
    return HigherBlockResult(
        **request.model_dump(),
        presentation=higher_block_presentation(request.shift, request.block_length),
    )


__all__ = [
    "compute_artin_mazur_zeta",
    "compute_block_language",
    "compute_higher_block",
    "compute_periodic_point_profile",
    "construct_finite_type_shift",
]
