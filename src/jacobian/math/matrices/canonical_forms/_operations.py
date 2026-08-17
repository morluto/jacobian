"""Domain adapter for exact canonical-form operations."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.canonical_forms import (
    characteristic_polynomial,
    invariant_factors,
    minimal_polynomial,
    primary_decomposition,
)
from jacobian.math.matrices.canonical_forms._models import (
    InvariantFactorEntry,
    MinimalPolynomialResult,
    MonicPolynomial,
    PrimaryDecompositionResult,
    RationalCanonicalFormResult,
    SquareMatrixRequest,
)


def _matrix_entries(
    request: SquareMatrixRequest,
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix.entries
    )


def _to_monic_polynomial(coefficients: Sequence[Fraction]) -> MonicPolynomial:
    return MonicPolynomial(
        coefficients=tuple(
            CanonicalRational.from_fraction(coefficient) for coefficient in coefficients
        )
    )


def compute_minimal_polynomial(
    request: SquareMatrixRequest,
) -> MinimalPolynomialResult:
    entries = _matrix_entries(request)
    minimal = minimal_polynomial(entries)
    characteristic = characteristic_polynomial(entries)
    return MinimalPolynomialResult(
        minimal_polynomial=_to_monic_polynomial(minimal),
        characteristic_polynomial=_to_monic_polynomial(characteristic),
        degree=len(minimal) - 1,
    )


def compute_rational_canonical_form(
    request: SquareMatrixRequest,
) -> RationalCanonicalFormResult:
    entries = _matrix_entries(request)
    factors = invariant_factors(entries)
    minimal = minimal_polynomial(entries)
    characteristic = characteristic_polynomial(entries)

    invariant_entries = tuple(
        InvariantFactorEntry(
            factor=_to_monic_polynomial(coefficients),
            block_size=len(coefficients) - 1,
        )
        for coefficients in factors
    )

    return RationalCanonicalFormResult(
        invariant_factors=invariant_entries,
        characteristic_polynomial=_to_monic_polynomial(characteristic),
        minimal_polynomial=_to_monic_polynomial(minimal),
        total_block_size=sum(entry.block_size for entry in invariant_entries),
    )


def compute_primary_decomposition(
    request: SquareMatrixRequest,
) -> PrimaryDecompositionResult:
    entries = _matrix_entries(request)
    components = primary_decomposition(entries)
    minimal = minimal_polynomial(entries)
    return PrimaryDecompositionResult(
        components=tuple(
            _to_monic_polynomial(coefficient) for coefficient in components
        ),
        minimal_polynomial=_to_monic_polynomial(minimal),
    )
