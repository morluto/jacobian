"""Weighted monotone endpoint profile kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.weighted_monotone_profiles._models import (
    WeightedMonotoneProfileResult,
)

__all__ = ["compute_weighted_monotone_profiles"]


def compute_weighted_monotone_profiles(
    alphabet: tuple[int, ...],
    weights: tuple[CanonicalRational, ...],
) -> WeightedMonotoneProfileResult:
    """Return the two exact endpoint DP profiles.

    S_i = w_i + max_{j<i, a_j <= a_i} S_j  (increasing)
    T_i = w_i + max_{j<i, a_j >= a_i} T_j  (decreasing)
    """
    n = len(alphabet)
    w = [wt.as_fraction() for wt in weights]

    increasing = [Fraction(0)] * n
    decreasing = [Fraction(0)] * n

    for i in range(n):
        best_inc = Fraction(0)
        best_dec = Fraction(0)

        for j in range(i):
            if alphabet[j] <= alphabet[i]:
                best_inc = max(best_inc, increasing[j])
            if alphabet[j] >= alphabet[i]:
                best_dec = max(best_dec, decreasing[j])

        increasing[i] = w[i] + best_inc
        decreasing[i] = w[i] + best_dec

    return WeightedMonotoneProfileResult(
        alphabet=alphabet,
        weights=weights,
        increasing_profile=tuple(
            CanonicalRational.from_fraction(v) for v in increasing
        ),
        decreasing_profile=tuple(
            CanonicalRational.from_fraction(v) for v in decreasing
        ),
    )
