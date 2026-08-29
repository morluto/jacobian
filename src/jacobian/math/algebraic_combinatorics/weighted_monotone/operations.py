"""Weighted monotone subsequence endpoint profile kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.algebraic_combinatorics.weighted_monotone._models import (
    EndpointProfileEntry,
    EndpointProfileResult,
    WeightedOrderedWord,
)

__all__ = ["compute_endpoint_profile"]


def compute_endpoint_profile(
    source: WeightedOrderedWord,
) -> EndpointProfileResult:
    """Return the two endpoint DP profiles for a weighted ordered word.

    S_i = w_i + max_{j<i, a_j <= a_i} S_j  (weakly increasing)
    T_i = w_i + max_{j<i, a_j >= a_i} T_j  (weakly decreasing)

    where the max over the empty set is 0.
    """
    word = source.word
    letters = list(word.letters)
    alphabet = list(word.alphabet)
    n = len(letters)
    weights = [w.as_fraction() for w in source.weights]

    letter_rank = {sym: i for i, sym in enumerate(alphabet)}

    s_values: list[Fraction] = []
    t_values: list[Fraction] = []

    for i in range(n):
        wi = weights[i]
        ri = letter_rank[letters[i]]

        s_best = Fraction(0)
        t_best = Fraction(0)
        for j in range(i):
            rj = letter_rank[letters[j]]
            if rj <= ri:
                s_best = max(s_best, s_values[j])
            if rj >= ri:
                t_best = max(t_best, t_values[j])

        s_values.append(wi + s_best)
        t_values.append(wi + t_best)

    entries = [
        EndpointProfileEntry(
            position=i,
            letter=letters[i],
            weight=source.weights[i],
            increasing_value=CanonicalRational.from_fraction(s_values[i]),
            decreasing_value=CanonicalRational.from_fraction(t_values[i]),
        )
        for i in range(n)
    ]

    return EndpointProfileResult(source=source, entries=tuple(entries))
