"""Domain-owned algebraic combinatorics operation adapters."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.algebraic_combinatorics import (
    conjugate_partition,
    hook_lengths,
    inverse_row_insertion_rsk,
    row_insertion_rsk,
    standard_young_tableaux_count,
)
from jacobian.math.algebraic_combinatorics._models import (
    ConjugatePartitionRequest,
    ConjugatePartitionResult,
    HookLengthRequest,
    HookLengthResult,
    RSKInverseWordRequest,
    RSKInverseWordResult,
    RSKPermutationRequest,
    RSKResult,
    RSKWordRequest,
    StandardYoungTableauCountRequest,
    StandardYoungTableauCountResult,
)
from jacobian.math.algebraic_combinatorics._rsk import _row_insert
from jacobian.math.algebraic_combinatorics.values import RSKTableauPair
from jacobian.math.symmetric_functions.values import IntegerPartition


def compute_hook_lengths(request: HookLengthRequest) -> HookLengthResult:
    parts = list(request.partition.parts)
    hooks = hook_lengths(parts)
    total_product = 1
    for row in hooks:
        for hook in row:
            total_product *= hook
    return HookLengthResult(
        hooks=tuple(tuple(row) for row in hooks),
        total_product=format_canonical_integer(total_product),
    )


def compute_syt_count(
    request: StandardYoungTableauCountRequest,
) -> StandardYoungTableauCountResult:
    parts = list(request.partition.parts)
    count = standard_young_tableaux_count(parts)
    n = sum(parts)
    return StandardYoungTableauCountResult(count=format_canonical_integer(count), n=n)


def compute_conjugate_partition(
    request: ConjugatePartitionRequest,
) -> ConjugatePartitionResult:
    parts = list(request.partition.parts)
    result = conjugate_partition(parts)
    return ConjugatePartitionResult(conjugate=IntegerPartition(parts=tuple(result)))


def compute_rsk_permutation(request: RSKPermutationRequest) -> RSKResult:
    """Compute the RSK correspondence for a permutation.

    Uses standard row insertion. P is the insertion tableau,
    Q is the recording tableau. The shape gives the partition.
    LIS length = first row length, LDS length = first column length.
    """
    perm = request.permutation

    p, q = _row_insert(perm)
    shape = tuple(len(row) for row in p)

    # LIS length = length of first row
    # LDS length = length of first column (number of rows)
    lis_length = len(p[0]) if p else 0
    lds_length = len(p)

    return RSKResult(
        p_tableau=p,
        q_tableau=q,
        shape=shape,
        lis_length=lis_length,
        lds_length=lds_length,
    )


def compute_rsk_word(request: RSKWordRequest) -> RSKTableauPair:
    """Compute and inverse-replay the compact ordinary-word RSK pair."""
    return row_insertion_rsk(request.word)


def compute_inverse_rsk_word(
    request: RSKInverseWordRequest,
) -> RSKInverseWordResult:
    """Reverse-insert and forward-replay one ordinary-word RSK pair."""
    return RSKInverseWordResult(word=inverse_row_insertion_rsk(request.pair))
