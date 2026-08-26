"""Exact canonical-form kernels and typed contracts over QQ."""

from jacobian.math.matrices.canonical_forms._models import (
    InvariantFactorEntry,
    MatrixPolynomialEvaluationResult,
    MinimalPolynomialResult,
    MonicPolynomial,
    PrimaryDecompositionResult,
    RationalCanonicalFormResult,
)
from jacobian.math.matrices.canonical_forms.operations import (
    characteristic_polynomial,
    invariant_factors,
    minimal_polynomial,
    primary_decomposition,
)

__all__ = [
    "InvariantFactorEntry",
    "MatrixPolynomialEvaluationResult",
    "MinimalPolynomialResult",
    "MonicPolynomial",
    "PrimaryDecompositionResult",
    "RationalCanonicalFormResult",
    "characteristic_polynomial",
    "invariant_factors",
    "minimal_polynomial",
    "primary_decomposition",
]
