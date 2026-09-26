"""Weighted monotone subsequence endpoint profile kernel."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian._execution import request_checkpoint
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.algebraic.weighted_monotone._models import (
    MAX_ENDPOINT_PROFILE_WORK,
    MAX_WEIGHTED_MONOTONE_ARITHMETIC_WORK,
    MAX_WEIGHTED_MONOTONE_COMPONENT_DIGITS,
    MAX_WEIGHTED_MONOTONE_RESULT_COMPONENT_DIGITS,
    MAX_WEIGHTED_MONOTONE_SOURCE_DIGITS,
    EndpointProfileEntry,
    EndpointProfileResult,
    WeightedMaximumResult,
    WeightedMonotonicity,
    WeightedOrderedWord,
)
from jacobian.math.logic.languages.words.values import FiniteWord

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


def _admit_weighted_maximum(source: WeightedOrderedWord) -> int:
    """Admit both DP work and a conservative exact-rational growth bound."""
    if not isinstance(source, WeightedOrderedWord):
        raise OperationDomainValidationError(
            location=("source",),
            code="weighted_word.invalid_source",
            message="source must be a WeightedOrderedWord value",
        )
    word = source.word
    if not isinstance(word, FiniteWord):
        raise OperationDomainValidationError(
            location=("source", "word"),
            code="weighted_word.invalid_source",
            message="source word must be a FiniteWord value",
        )
    n = len(word.letters)
    if n > 500 or len(source.weights) != n:
        raise OperationDomainValidationError(
            location=("source",),
            code="weighted_word.invalid_source",
            message="source word and rational weights must have equal length at most 500",
        )
    if len(set(word.alphabet)) != len(word.alphabet) or any(
        letter not in word.alphabet for letter in word.letters
    ):
        raise OperationDomainValidationError(
            location=("source", "word"),
            code="weighted_word.invalid_source",
            message="source must retain a canonical explicitly ordered word",
        )
    numerator_digits = 1
    denominator_lcm = 1
    source_digits = 0
    for weight in source.weights:
        if (
            not isinstance(weight, CanonicalRational)
            or weight.den <= 0
            or weight.num < 0
        ):
            raise OperationDomainValidationError(
                location=("source", "weights"),
                code="weighted_word.invalid_weight",
                message="every weight must be a nonnegative canonical rational",
            )
        if max(weight.num.bit_length(), weight.den.bit_length()) > 4 * (
            MAX_WEIGHTED_MONOTONE_COMPONENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("source", "weights"),
                code="weighted_word.component_digits",
                message="rational components exceed the admitted 256-digit envelope",
            )
        num_digits = decimal_digit_width(weight.num)
        den_digits = decimal_digit_width(weight.den)
        if max(num_digits, den_digits) > MAX_WEIGHTED_MONOTONE_COMPONENT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("source", "weights"),
                code="weighted_word.component_digits",
                message="rational components exceed the admitted 256-digit envelope",
            )
        source_digits += num_digits + den_digits
        if source_digits > MAX_WEIGHTED_MONOTONE_SOURCE_DIGITS:
            raise OperationResourceAdmissionError(
                location=("source", "weights"),
                code="weighted_word.source_digits",
                message="combined rational source exceeds the 4,096-digit envelope",
            )
        numerator_digits = max(numerator_digits, num_digits)
        if weight.den != 1:
            denominator_lcm = (
                denominator_lcm // gcd(denominator_lcm, weight.den) * weight.den
            )

    # A common denominator is the LCM of distinct input denominators. Repeated
    # equal denominators do not multiply growth; use their product as a safe
    # bound while retaining this important common-denominator case.
    denominator_lcm_digits = (
        decimal_digit_width(denominator_lcm) if denominator_lcm != 1 else 0
    )
    # Every witness sum is at most n times the largest input numerator over
    # the product of distinct nonunit input denominators. This bounds both Fraction
    # operands before any dynamic-programming state is materialized.
    carry_digits = decimal_digit_width(max(n, 1))
    growth_digits = numerator_digits + denominator_lcm_digits + carry_digits
    if growth_digits > MAX_WEIGHTED_MONOTONE_RESULT_COMPONENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("source", "weights"),
            code="weighted_word.result_growth_exceeded",
            message=(
                "exact witness sums may exceed the admitted "
                f"{MAX_WEIGHTED_MONOTONE_RESULT_COMPONENT_DIGITS}-digit result bound"
            ),
        )
    comparisons = n * max(n - 1, 0) // 2
    rational_cost = growth_digits * growth_digits
    # Each pair may compare two rationals; each vertex adds one weight and
    # canonicalizes the final exact sum. The factors cover schoolbook bigint
    # multiplication/division and Euclidean gcd work in Fraction arithmetic.
    arithmetic_work = comparisons * 2 * rational_cost + n * 12 * rational_cost
    if arithmetic_work > MAX_WEIGHTED_MONOTONE_ARITHMETIC_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "weights"),
            code="weighted_word.arithmetic_work_exceeded",
            message=(
                "quadratic exact-rational work exceeds the admitted "
                f"{MAX_WEIGHTED_MONOTONE_ARITHMETIC_WORK}-unit bound"
            ),
        )
    return growth_digits


def _maximum_weight_monotone_subsequence(
    source: WeightedOrderedWord,
    monotonicity: WeightedMonotonicity,
) -> WeightedMaximumResult:
    _admit_weighted_maximum(source)
    word = source.word
    n = len(word.letters)
    ranks = {symbol: rank for rank, symbol in enumerate(word.alphabet)}
    weights = [weight.as_fraction() for weight in source.weights]
    values: list[Fraction] = []
    predecessor = [-1] * n

    for end in range(n):
        request_checkpoint("during weighted monotone-subsequence dynamic programming")
        best_prefix = Fraction(0)
        best_predecessor = -1
        end_rank = ranks[word.letters[end]]
        for previous in range(end):
            previous_rank = ranks[word.letters[previous]]
            follows = (
                previous_rank <= end_rank
                if monotonicity == "NONDECREASING"
                else previous_rank >= end_rank
            )
            if follows and values[previous] > best_prefix:
                best_prefix = values[previous]
                best_predecessor = previous
        values.append(weights[end] + best_prefix)
        predecessor[end] = best_predecessor

    best_end = -1
    for end, value in enumerate(values):
        if best_end < 0 or value > values[best_end]:
            best_end = end
    if best_end < 0:
        total = Fraction(0)
        indices: tuple[int, ...] = ()
    else:
        total = values[best_end]
        reversed_indices: list[int] = []
        cursor = best_end
        while cursor >= 0:
            reversed_indices.append(cursor)
            cursor = predecessor[cursor]
        indices = tuple(reversed(reversed_indices))

    return WeightedMaximumResult._from_kernel(
        source,
        monotonicity,
        CanonicalRational.from_fraction(total),
        indices,
        tuple(word.letters[position] for position in indices),
    )


def maximum_weight_nondecreasing_subsequence(
    source: WeightedOrderedWord,
) -> WeightedMaximumResult:
    """Return the maximum exact weight of a weakly increasing subsequence."""
    return _maximum_weight_monotone_subsequence(source, "NONDECREASING")


def maximum_weight_nonincreasing_subsequence(
    source: WeightedOrderedWord,
) -> WeightedMaximumResult:
    """Return the maximum exact weight of a weakly decreasing subsequence."""
    return _maximum_weight_monotone_subsequence(source, "NONINCREASING")


__all__ = [
    "compute_endpoint_profile",
    "maximum_weight_nondecreasing_subsequence",
    "maximum_weight_nonincreasing_subsequence",
]
