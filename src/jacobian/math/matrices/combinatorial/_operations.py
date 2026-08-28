"""Domain adapter for combinatorial-matrix operations."""

from __future__ import annotations

from jacobian.math.matrices.combinatorial._models import (
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
from jacobian.math.matrices.combinatorial.operations import (
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
    return sign_profile(request.matrix)


def compute_gram_profile(request: GramProfileRequest) -> GramProfileResult:
    return gram_profile(request.matrix)


def compute_normalize(request: NormalizeRequest) -> NormalizeResult:
    return normalize(request.matrix)


def compute_determinant_profile(
    request: DeterminantProfileRequest,
) -> DeterminantProfileResult:
    return determinant_profile(request.matrix)


def compute_sylvester(request: SylvesterRequest) -> SylvesterResult:
    return sylvester(request.k)
