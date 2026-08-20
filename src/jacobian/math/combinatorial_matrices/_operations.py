"""Domain adapter for combinatorial-matrix operations."""

from __future__ import annotations

from jacobian.math.combinatorial_matrices._models import (
    DeterminantProfileRequest,
    DeterminantProfileResult,
    GramProfileRequest,
    GramProfileResult,
    NormalizeRequest,
    NormalizeResult,
    SignProfileRequest,
    SignProfileResult,
    SylvesterRequest,
    SylvesterResult,
)
from jacobian.math.combinatorial_matrices.operations import (
    determinant_profile,
    gram_profile,
    normalize,
    sign_profile,
    sylvester,
)

__all__ = [
    "compute_determinant_profile",
    "compute_gram_profile",
    "compute_normalize",
    "compute_sign_profile",
    "compute_sylvester",
]


def compute_sign_profile(request: SignProfileRequest) -> SignProfileResult:
    result = sign_profile(request.matrix)
    return SignProfileResult(
        row_count=result["row_count"],  # type: ignore[arg-type]
        column_count=result["column_count"],  # type: ignore[arg-type]
        plus_one_count=result["plus_one_count"],  # type: ignore[arg-type]
        minus_one_count=result["minus_one_count"],  # type: ignore[arg-type]
        row_sums=result["row_sums"],  # type: ignore[arg-type]
        column_sums=result["column_sums"],  # type: ignore[arg-type]
        is_square=result["is_square"],  # type: ignore[arg-type]
    )


def compute_gram_profile(request: GramProfileRequest) -> GramProfileResult:
    result = gram_profile(request.matrix)
    return GramProfileResult(
        order=result["order"],  # type: ignore[arg-type]
        gram=result["gram"],  # type: ignore[arg-type]
        diagonal_residuals=result["diagonal_residuals"],  # type: ignore[arg-type]
        nonzero_off_diagonal=result["nonzero_off_diagonal"],  # type: ignore[arg-type]
        is_hadamard=result["is_hadamard"],  # type: ignore[arg-type]
    )


def compute_normalize(request: NormalizeRequest) -> NormalizeResult:
    result = normalize(request.matrix)
    return NormalizeResult(
        normalized=result["normalized"],  # type: ignore[arg-type]
        row_switches=result["row_switches"],  # type: ignore[arg-type]
        column_switches=result["column_switches"],  # type: ignore[arg-type]
    )


def compute_determinant_profile(
    request: DeterminantProfileRequest,
) -> DeterminantProfileResult:
    result = determinant_profile(request.matrix)
    return DeterminantProfileResult(
        order=result["order"],  # type: ignore[arg-type]
        determinant_magnitude=result["determinant_magnitude"],  # type: ignore[arg-type]
        gram_determinant=result["gram_determinant"],  # type: ignore[arg-type]
        identity=result["identity"],  # type: ignore[arg-type]
    )


def compute_sylvester(request: SylvesterRequest) -> SylvesterResult:
    result = sylvester(request.k)
    return SylvesterResult(
        matrix=result["matrix"],  # type: ignore[arg-type]
        construction=result["construction"],  # type: ignore[arg-type]
        order=result["order"],  # type: ignore[arg-type]
    )
