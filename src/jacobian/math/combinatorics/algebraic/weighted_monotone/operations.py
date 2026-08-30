"""Weighted monotone subsequence endpoint profile kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic.weighted_monotone._models import (
    MAX_ENDPOINT_PROFILE_WORK,
    EndpointProfileEntry,
    EndpointProfileResult,
    WeightedOrderedWord,
)

__all__ = ["compute_endpoint_profile"]


def _rational_size(digits: int) -> int:
    scalar = "9" * max(1, digits)
    return strict_json_object_size(
        (
            ("num", len(encode_strict_json(scalar))),
            ("den", len(encode_strict_json(scalar))),
        )
    )


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
    entry_sizes = []
    cumulative_digits = 0
    for index, weight in enumerate(source.weights):
        cumulative_digits += canonical_rational_component_digits(weight)
        entry_sizes.append(
            strict_json_object_size(
                (
                    ("position", len(encode_strict_json(index))),
                    ("letter", len(encode_strict_json(source.word.letters[index]))),
                    (
                        "weight",
                        len(encode_strict_json(weight.model_dump(mode="json"))),
                    ),
                    (
                        "increasing_value",
                        _rational_size(cumulative_digits + len(str(index + 1))),
                    ),
                    (
                        "decreasing_value",
                        _rational_size(cumulative_digits + len(str(index + 1))),
                    ),
                )
            )
        )
    result_bytes = strict_json_object_size(
        (
            (
                "source",
                len(encode_strict_json(source.model_dump(mode="json"))),
            ),
            ("entries", 2 + max(n - 1, 0) + sum(entry_sizes)),
        )
    )
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("source",),
            code="weighted_word.result_too_large",
            message="endpoint profile exceeds the canonical result envelope",
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
