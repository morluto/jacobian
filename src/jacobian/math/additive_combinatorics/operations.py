"""Native exact kernels for additive combinatorics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from jacobian.canonical import format_canonical_integer
from jacobian.math.additive_combinatorics.values import (
    MAX_SUBSET_SUM_PROFILE_ENTRIES,
    IndexedIntegerSequence,
    SubsetSumProfile,
)

MAX_SUBSET_SUM_DP_TRANSITIONS = 4_000_000
MAX_SUBSET_SUM_PROFILE_RESULT_BYTES = 4 * 1024 * 1024
_SUBSET_SUM_DP_PASSES = 2


@dataclass(frozen=True)
class _SubsetSumProfileEnvelope:
    """Conservative pre-execution bounds for one complete profile."""

    support_bound: int
    transition_bound: int
    maximum_sum_characters: int
    maximum_multiplicity_digits: int
    result_byte_bound: int


def _subset_sum_profile_envelope(
    source: IndexedIntegerSequence,
) -> _SubsetSumProfileEnvelope:
    """Derive complete-profile work, intermediate, and output bounds.

    For values grouped by equality, a subset is determined numerically by how
    many copies of each nonzero value it selects. Thus the support is bounded
    simultaneously by ``2^n``, the integral sum span, and the product of one
    plus each nonzero value's multiplicity. Every intermediate DP support is a
    subset of the final support because the exclude branch retains old sums.
    """

    values = source.as_int_tuple()
    item_count = len(values)
    total_subsets = 1 << item_count

    negative_sum = sum(value for value in values if value < 0)
    positive_sum = sum(value for value in values if value > 0)
    span_bound = positive_sum - negative_sum + 1

    grouped = Counter(value for value in values if value != 0)
    selection_vector_bound = math.prod(
        multiplicity + 1 for multiplicity in grouped.values()
    )
    support_bound = min(total_subsets, span_bound, selection_vector_bound)
    # The public result validator independently replays the DP so a forged or
    # truncated exact profile cannot revalidate. Charge both the construction
    # and validation passes, with two dictionary updates per retained state.
    transition_bound = _SUBSET_SUM_DP_PASSES * 2 * item_count * support_bound

    maximum_sum_characters = max(
        len(format_canonical_integer(negative_sum)),
        len(format_canonical_integer(positive_sum)),
    )
    maximum_multiplicity_digits = len(format_canonical_integer(total_subsets))

    source_wire_bound = 64 + sum(len(item) + 3 for item in source.items)
    entry_wire_bound = 96 + maximum_sum_characters + maximum_multiplicity_digits
    result_byte_bound = 1024 + source_wire_bound + support_bound * entry_wire_bound

    if support_bound > MAX_SUBSET_SUM_PROFILE_ENTRIES:
        raise ValueError(
            "predicted subset-sum support exceeds the "
            f"{MAX_SUBSET_SUM_PROFILE_ENTRIES}-entry profile bound"
        )
    if transition_bound > MAX_SUBSET_SUM_DP_TRANSITIONS:
        raise ValueError(
            "predicted subset-sum DP transitions exceed the "
            f"{MAX_SUBSET_SUM_DP_TRANSITIONS}-transition work bound"
        )
    if result_byte_bound > MAX_SUBSET_SUM_PROFILE_RESULT_BYTES:
        raise ValueError(
            "predicted subset-sum result exceeds the "
            f"{MAX_SUBSET_SUM_PROFILE_RESULT_BYTES}-byte result bound"
        )

    return _SubsetSumProfileEnvelope(
        support_bound=support_bound,
        transition_bound=transition_bound,
        maximum_sum_characters=maximum_sum_characters,
        maximum_multiplicity_digits=maximum_multiplicity_digits,
        result_byte_bound=result_byte_bound,
    )


def _subset_sum_profile_counts(source: IndexedIntegerSequence) -> dict[int, int]:
    """Return exact counts for every indexed subset sum, including empty."""

    counts: dict[int, int] = {0: 1}
    for value in source.as_int_tuple():
        next_counts: dict[int, int] = {}
        for subtotal, multiplicity in counts.items():
            next_counts[subtotal] = next_counts.get(subtotal, 0) + multiplicity
            included = subtotal + value
            next_counts[included] = next_counts.get(included, 0) + multiplicity
        counts = next_counts
    return counts


def subset_sum_profile(source: IndexedIntegerSequence) -> SubsetSumProfile:
    """Return the complete indexed-subset multiplicity profile of ``source``.

    Every position is independently selected at most once. Equal values and
    zeros therefore contribute separate multiplicity even though they may not
    enlarge the numeric support. The empty subset is included by definition.
    """

    envelope = _subset_sum_profile_envelope(source)
    counts = _subset_sum_profile_counts(source)
    if len(counts) > envelope.support_bound:
        raise RuntimeError("subset-sum support exceeded its admitted bound")
    return SubsetSumProfile.from_counts(source, counts)


__all__ = ["subset_sum_profile"]
