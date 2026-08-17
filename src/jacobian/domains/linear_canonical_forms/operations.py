"""Domain adapter for linear canonical-form operations."""

from __future__ import annotations

from typing import Any

from jacobian.contracts.canonical_forms import (
    InvariantFactorEntry,
    MinimalPolynomialResult,
    MonicPolynomial,
    PrimaryDecompositionResult,
    RationalCanonicalFormResult,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.math.canonical_forms import (
    characteristic_polynomial,
    invariant_factors,
    minimal_polynomial,
    primary_decomposition,
)


def _matrix_entries(request: Any) -> list[list[Any]]:
    return [[entry.as_fraction() for entry in row] for row in request.matrix.entries]


def _to_monic_polynomial(coeffs: list[Any]) -> MonicPolynomial:
    return MonicPolynomial(
        coefficients=tuple(
            CanonicalRational.from_fraction(_to_fraction(c)) for c in coeffs
        )
    )


def _to_fraction(value: Any) -> Any:
    from fractions import Fraction

    if isinstance(value, (int, float)):
        return Fraction(value)
    # SymPy Rational
    return Fraction(int(value.p), int(value.q))


def compute_minimal_polynomial(request: Any) -> MinimalPolynomialResult:
    entries = _matrix_entries(request)
    mp_coeffs = minimal_polynomial(entries)
    cp_coeffs = characteristic_polynomial(entries)

    degree = len(mp_coeffs) - 1
    return MinimalPolynomialResult(
        minimal_polynomial=_to_monic_polynomial(mp_coeffs),
        characteristic_polynomial=_to_monic_polynomial(cp_coeffs),
        degree=degree,
    )


def compute_rational_canonical_form(request: Any) -> RationalCanonicalFormResult:
    entries = _matrix_entries(request)
    ifs = invariant_factors(entries)
    mp_coeffs = minimal_polynomial(entries)
    cp_coeffs = characteristic_polynomial(entries)

    invariant_entries = tuple(
        InvariantFactorEntry(
            factor=_to_monic_polynomial(coeffs),
            block_size=len(coeffs) - 1,
        )
        for coeffs in ifs
    )

    total_block_size = sum(entry.block_size for entry in invariant_entries)

    return RationalCanonicalFormResult(
        invariant_factors=invariant_entries,
        characteristic_polynomial=_to_monic_polynomial(cp_coeffs),
        minimal_polynomial=_to_monic_polynomial(mp_coeffs),
        total_block_size=total_block_size,
    )


def compute_primary_decomposition(request: Any) -> PrimaryDecompositionResult:
    entries = _matrix_entries(request)
    pd = primary_decomposition(entries)
    mp_coeffs = minimal_polynomial(entries)

    components = tuple(_to_monic_polynomial(coeffs) for coeffs in pd)

    return PrimaryDecompositionResult(
        components=components,
        minimal_polynomial=_to_monic_polynomial(mp_coeffs),
    )
