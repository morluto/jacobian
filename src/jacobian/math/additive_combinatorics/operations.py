"""Native exact kernels for additive combinatorics."""

from __future__ import annotations

from jacobian.math.additive_combinatorics import _subset_sum_profile
from jacobian.math.additive_combinatorics._subset_sum_profile import (
    subset_sum_profile_counts,
    subset_sum_profile_envelope,
)
from jacobian.math.additive_combinatorics.values import (
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


# Compatibility aliases remain private while callers move to the owner-local
# admission helper; they are not used by result construction.
_subset_sum_profile_envelope = subset_sum_profile_envelope
_subset_sum_profile_counts = subset_sum_profile_counts
MAX_SUBSET_SUM_DP_TRANSITIONS = _subset_sum_profile.MAX_SUBSET_SUM_DP_TRANSITIONS
MAX_SUBSET_SUM_PROFILE_RESULT_BYTES = (
    _subset_sum_profile.MAX_SUBSET_SUM_PROFILE_RESULT_BYTES
)

__all__ = ["subset_sum_profile"]
