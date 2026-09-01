"""Weighted monotone subsequence endpoint profile kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic.weighted_monotone._models import (
    MAX_ENDPOINT_PROFILE_WORK,
    EndpointProfileEntry,
    EndpointProfileResult,
    WeightedOrderedWord,
)

__all__ = ["compute_endpoint_profile"]


def _admit_endpoint_profile(source: WeightedOrderedWord) -> None:
    if not isinstance(source, WeightedOrderedWord):
        raise OperationDomainValidationError(
            location=("source",),
            code="weighted_word.invalid_source",
            message="source must be a WeightedOrderedWord value",
        )
    n = len(source.word.letters)
    if n * max(n - 1, 0) > MAX_ENDPOINT_PROFILE_WORK:
        raise OperationDomainValidationError(
            location=("source", "word", "letters"),
            code="weighted_word.work_bound_exceeded",
            message="the quadratic endpoint-profile work envelope is exceeded",
        )
    max_digits = max(
        (canonical_rational_component_digits(weight) for weight in source.weights),
        default=1,
    )
    cumulative_carry_digits = len(str(max(n, 1)))
    if n * max_digits + cumulative_carry_digits > 32_768:
        raise OperationDomainValidationError(
            location=("source", "weights"),
            code="weighted_word.result_growth_exceeded",
            message="endpoint rational growth exceeds the canonical digit envelope",
        )


def compute_endpoint_profile(
    source: WeightedOrderedWord,
) -> EndpointProfileResult:
    """Return the two endpoint DP profiles for a weighted ordered word.

    S_i = w_i + max_{j<i, a_j <= a_i} S_j  (weakly increasing)
    T_i = w_i + max_{j<i, a_j >= a_i} T_j  (weakly decreasing)

    where the max over the empty set is 0.
    """
    _admit_endpoint_profile(source)
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
