"""Rational subset-sum profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.additive.rational_subset_sum._models import (
    RationalSubsetSumResult,
    SubsetSumRow,
)

__all__ = ["compute_rational_subset_sum_profile"]


def compute_rational_subset_sum_profile(
    values: tuple[CanonicalRational, ...],
) -> RationalSubsetSumResult:
    """Return the complete profile of all subset sums of the given rationals.

    For every subset I of the source indices, compute the sum of the
    corresponding rational values. Group by canonical rational value
    and count multiplicities.
    """
    fractions = [v.as_fraction() for v in values]
    n = len(fractions)
    sum_to_count: dict[Fraction, int] = {}

    for r in range(n + 1):
        for indices in combinations(range(n), r):
            s = sum(fractions[i] for i in indices)
            sum_to_count[s] = sum_to_count.get(s, 0) + 1

    rows = [
        SubsetSumRow(
            sum_value=CanonicalRational.from_fraction(s),
            multiplicity=count,
        )
        for s, count in sorted(sum_to_count.items())
    ]
    return RationalSubsetSumResult(
        values=values,
        rows=tuple(rows),
    )
