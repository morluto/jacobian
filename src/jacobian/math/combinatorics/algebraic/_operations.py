"""Domain-owned algebraic combinatorics operation adapters."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic import (
    conjugate_partition,
    hook_lengths,
    inverse_row_insertion_rsk,
    row_insertion_rsk,
    standard_young_tableaux_count,
)
from jacobian.math.combinatorics.algebraic._models import (
    ConjugatePartitionRequest,
    ConjugatePartitionResult,
    HookLengthRequest,
    HookLengthResult,
    RSKInverseWordRequest,
    RSKPermutationRequest,
    RSKResult,
    RSKWordRequest,
    StandardYoungTableauCountRequest,
    StandardYoungTableauCountResult,
)
from jacobian.math.combinatorics.algebraic._rsk import _row_insert
from jacobian.math.combinatorics.algebraic.values import RSKTableauPair
from jacobian.math.logic.languages.words.values import FiniteWord


def compute_hook_lengths(request: HookLengthRequest) -> HookLengthResult:
    hooks = hook_lengths(request.partition)
    total_product = 1
    for row in hooks:
        for hook in row:
            total_product *= hook
    return HookLengthResult(
        hooks=hooks,
        total_product=format_canonical_integer(total_product),
    )


def compute_syt_count(
    request: StandardYoungTableauCountRequest,
) -> StandardYoungTableauCountResult:
    count = standard_young_tableaux_count(request.partition)
    n = sum(request.partition.parts)
    return StandardYoungTableauCountResult(count=format_canonical_integer(count), n=n)


def compute_conjugate_partition(
    request: ConjugatePartitionRequest,
) -> ConjugatePartitionResult:
    return ConjugatePartitionResult(conjugate=conjugate_partition(request.partition))


def compute_rsk_permutation(request: RSKPermutationRequest) -> RSKResult:
    """Compute the RSK correspondence for a permutation.

    Uses standard row insertion. P is the insertion tableau,
    Q is the recording tableau. The shape gives the partition.
    LIS length = first row length, LDS length = first column length.
    """
    if sorted(request.permutation) != list(range(1, len(request.permutation) + 1)):
        raise OperationDomainValidationError(
            location=("permutation",),
            code="algebraic_combinatorics.permutation_invalid",
            message="permutation must be a permutation of 1..n",
        )
    insertion_rows, recording_rows = _row_insert(request.permutation)
    return RSKResult._from_kernel(
        request,
        insertion_rows=insertion_rows,
        recording_rows=recording_rows,
    )


def compute_rsk_word(request: RSKWordRequest) -> RSKTableauPair:
    """Compute the compact ordinary-word RSK pair."""
    return row_insertion_rsk(request.word)


def compute_inverse_rsk_word(
    request: RSKInverseWordRequest,
) -> FiniteWord:
    """Reverse-insert one ordinary-word RSK pair."""
    return inverse_row_insertion_rsk(request.pair)
