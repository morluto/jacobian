"""Rational subset-sum profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.rational_subset_sum._models import (
    RationalSubsetSumEntry,
    RationalSubsetSumResult,
    require_rational_subset_sum_envelope,
)

__all__ = ["compute_rational_subset_sum_profile"]


def compute_rational_subset_sum_profile(
    values: tuple[CanonicalRational, ...],
) -> RationalSubsetSumResult:
    """Return every attainable subset sum and its multiplicity.

    For each subset I of {0,...,n-1}, the sum is sum(values[i] for i in I).
    The multiplicity of a sum s is the number of subsets achieving that sum.
    """
    try:
        require_rational_subset_sum_envelope(values)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("values",),
            code="rational_subset_sum.envelope_exceeded",
            message=str(exc),
        ) from exc
    n = len(values)
    fracs = [v.as_fraction() for v in values]

    sums: dict[Fraction, int] = {}
    sums[Fraction(0)] = 1  # Empty subset

    if n > 0:
        for size in range(1, n + 1):
            for subset in combinations(range(n), size):
                s = sum((fracs[i] for i in subset), Fraction(0))
                sums[s] = sums.get(s, 0) + 1

    if any(
        len(format_canonical_integer(component).lstrip("-"))
        > MAX_CANONICAL_RATIONAL_DIGITS
        for subset_sum in sums
        for component in (subset_sum.numerator, subset_sum.denominator)
    ):
        raise OperationDomainValidationError(
            location=("values",),
            code="rational_subset_sum.derived_rational_bound",
            message=(
                "a derived subset sum exceeds the canonical rational "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
            ),
        )

    entries = tuple(
        RationalSubsetSumEntry(
            sum=CanonicalRational.from_fraction(s),
            multiplicity=m,
        )
        for s, m in sorted(sums.items())
    )

    return RationalSubsetSumResult(
        values=values,
        entries=entries,
        support_cardinality=len(sums),
    )
