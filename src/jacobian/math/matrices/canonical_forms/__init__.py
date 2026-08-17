"""Exact canonical-form kernels and typed contracts over QQ."""

from jacobian.math.matrices.canonical_forms._models import (
    InvariantFactorEntry,
    MinimalPolynomialResult,
    MonicPolynomial,
    PrimaryDecompositionResult,
    RationalCanonicalFormResult,
    SquareMatrixRequest,
)
from jacobian.math.matrices.canonical_forms.operations import (
    characteristic_polynomial,
    invariant_factors,
    minimal_polynomial,
    primary_decomposition,
)

__all__ = [
    "InvariantFactorEntry",
    "MinimalPolynomialResult",
    "MonicPolynomial",
    "PrimaryDecompositionResult",
    "RationalCanonicalFormResult",
    "SquareMatrixRequest",
    "characteristic_polynomial",
    "invariant_factors",
    "minimal_polynomial",
    "primary_decomposition",
]
