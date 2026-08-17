"""Domain-owned algebraic combinatorics operation adapters."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.algebraic_combinatorics import (
    conjugate_partition,
    hook_lengths,
    standard_young_tableaux_count,
)
from jacobian.math.algebraic_combinatorics._models import (
    ConjugatePartitionRequest,
    ConjugatePartitionResult,
    HookLengthRequest,
    HookLengthResult,
    StandardYoungTableauCountRequest,
    StandardYoungTableauCountResult,
)


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
    return ConjugatePartitionResult(conjugate=tuple(result))
