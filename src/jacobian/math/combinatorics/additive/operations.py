"""Native exact kernels for additive combinatorics."""

from __future__ import annotations

from jacobian.math.combinatorics.additive._subset_sum_profile import (
    subset_sum_profile_counts,
    subset_sum_profile_envelope,
)
from jacobian.math.combinatorics.additive.values import (
    IndexedIntegerSequence,
    SubsetSumProfile,
)


def subset_sum_profile(source: IndexedIntegerSequence) -> SubsetSumProfile:
    """Return the complete indexed-subset multiplicity profile of ``source``.

    Every position is independently selected at most once. Equal values and
    zeros therefore contribute separate multiplicity even though they may not
    enlarge the numeric support. The empty subset is included by definition.
    """

    envelope = subset_sum_profile_envelope(source)
    counts = subset_sum_profile_counts(source)
    if len(counts) > envelope.support_bound:
        raise RuntimeError("subset-sum support exceeded its admitted bound")
    return SubsetSumProfile._from_kernel(source, counts)


__all__ = ["subset_sum_profile"]
