"""Wire adapters for public symbolic dynamics operations."""

from __future__ import annotations

from jacobian.math.dynamics.symbolic._models import (
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
from jacobian.math.dynamics.symbolic.operations import (
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
    determinant, zeta = artin_mazur_zeta(request.shift)
    return _from_kernel_artin_mazur_zeta(
        request,
        determinant,
        zeta,
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


__all__ = [
    "compute_artin_mazur_zeta",
    "compute_block_language",
    "compute_higher_block",
    "compute_periodic_point_profile",
    "construct_finite_type_shift",
]
