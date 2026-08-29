"""Rational fixed-arity sum profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.additive.rational_fixed_arity._models import (
    RationalFixedAritySumResult,
    SumProfileRow,
)

__all__ = ["compute_rational_fixed_arity_sum_profile"]


def compute_rational_fixed_arity_sum_profile(
    values: tuple[CanonicalRational, ...],
    arity: int,
) -> RationalFixedAritySumResult:
    """Return the complete profile of sums of exactly arity distinct indexed values.

    For each strictly increasing index h-tuple, compute the sum of the
    corresponding rational values. Group by canonical rational value and
    count multiplicities.
    """
    fractions = [v.as_fraction() for v in values]
    sum_to_count: dict[Fraction, int] = {}

    for indices in combinations(range(len(values)), arity):
        total = sum(fractions[i] for i in indices)
        sum_to_count[total] = sum_to_count.get(total, 0) + 1

    rows = [
        SumProfileRow(
            sum_value=CanonicalRational.from_fraction(s),
            multiplicity=count,
        )
        for s, count in sorted(sum_to_count.items())
    ]
    return RationalFixedAritySumResult(
        values=values,
        arity=arity,
        rows=tuple(rows),
    )
