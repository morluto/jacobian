"""Wire adapters for public symbolic dynamics operations."""

from __future__ import annotations

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
    _from_kernel_artin_mazur_zeta,
    _from_kernel_block_language,
    _from_kernel_finite_type_shift,
    _from_kernel_higher_block,
    _from_kernel_periodic_point_profile,
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
    return _from_kernel_artin_mazur_zeta(
        request,
        determinant,
        zeta,
        coefficients,
    )


def construct_finite_type_shift(
    request: FiniteTypeShiftRequest,
) -> FiniteTypeShiftResult:
    return _from_kernel_finite_type_shift(
        request,
        finite_type_presentation(request.shift),
        normalize_forbidden_blocks(request.shift),
    )


def compute_block_language(request: BlockLanguageRequest) -> BlockLanguageResult:
    allowed = block_language(request.shift, request.block_length)
    return _from_kernel_block_language(request, allowed)


def compute_periodic_point_profile(
    request: PeriodicPointProfileRequest,
) -> PeriodicPointProfileResult:
    fixed, exact, orbits = periodic_point_profile(request.shift, request.max_period)
    return _from_kernel_periodic_point_profile(request, fixed, exact, orbits)


def compute_higher_block(request: HigherBlockRequest) -> HigherBlockResult:
    return _from_kernel_higher_block(
        request, higher_block_presentation(request.shift, request.block_length)
    )


def verify_finite_type_shift_result(result: FiniteTypeShiftResult) -> bool:
    """Verify an independently supplied finite-type presentation claim."""

    return result.presentation == finite_type_presentation(
        result.shift
    ) and result.normalized_forbidden_blocks == normalize_forbidden_blocks(result.shift)


def verify_block_language_result(result: BlockLanguageResult) -> bool:
    """Verify an independently supplied complete block-language claim."""

    return result.allowed_blocks == block_language(result.shift, result.block_length)


def verify_periodic_point_profile_result(result: PeriodicPointProfileResult) -> bool:
    """Verify an independently supplied trace-and-Möbius profile claim."""

    fixed, exact, orbits = periodic_point_profile(result.shift, result.max_period)
    expected = _from_kernel_periodic_point_profile(
        PeriodicPointProfileRequest(shift=result.shift, max_period=result.max_period),
        fixed,
        exact,
        orbits,
    )
    return result == expected


def verify_higher_block_result(result: HigherBlockResult) -> bool:
    """Verify an independently supplied higher-block presentation claim."""

    return result.presentation == higher_block_presentation(
        result.shift, result.block_length
    )


def verify_artin_mazur_zeta_result(result: ArtinMazurZetaResult) -> bool:
    """Verify an independently supplied zeta claim within its request bound."""

    determinant, zeta, coefficients = artin_mazur_zeta(
        result.shift, result.replay_period
    )
    expected = _from_kernel_artin_mazur_zeta(
        ArtinMazurZetaRequest(shift=result.shift, replay_period=result.replay_period),
        determinant,
        zeta,
        coefficients,
    )
    return result == expected


__all__ = [
    "compute_artin_mazur_zeta",
    "compute_block_language",
    "compute_higher_block",
    "compute_periodic_point_profile",
    "construct_finite_type_shift",
    "verify_artin_mazur_zeta_result",
    "verify_block_language_result",
    "verify_finite_type_shift_result",
    "verify_higher_block_result",
    "verify_periodic_point_profile_result",
]
