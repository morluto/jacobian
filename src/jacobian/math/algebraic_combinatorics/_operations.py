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
    RSKPermutationRequest,
    RSKResult,
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


def compute_rsk_permutation(request: RSKPermutationRequest) -> RSKResult:
    """Compute the RSK correspondence for a permutation.

    Uses standard row insertion. P is the insertion tableau,
    Q is the recording tableau. The shape gives the partition.
    LIS length = first row length, LDS length = first column length.
    """
    perm = request.permutation

    if not perm:
        return RSKResult(
            p_tableau=(),
            q_tableau=(),
            shape=(),
            lis_length=0,
            lds_length=0,
        )

    # P and Q tableaux as lists of lists
    p: list[list[int]] = []
    q: list[list[int]] = []

    for idx, value in enumerate(perm):
        # Row insertion of value into tableau
        current = value
        row_idx = 0

        while row_idx < len(p):
            row = p[row_idx]
            # Find first element >= current in this row
            insert_pos = len(row)
            for i, v in enumerate(row):
                if v >= current:
                    insert_pos = i
                    break
            if insert_pos < len(row):
                # Bump
                bumped = row[insert_pos]
                row[insert_pos] = current
                current = bumped
                row_idx += 1
            else:
                # Append to end of row
                row.append(current)
                # Record in Q
                if row_idx >= len(q):
                    q.append([])
                q[row_idx].append(idx + 1)
                row_idx = -1  # Signal we inserted
                break

        if row_idx != -1 and row_idx >= len(p):
            # New row
            p.append([current])
            if row_idx >= len(q):
                q.append([])
            q[row_idx].append(idx + 1)

    shape = tuple(len(row) for row in p)

    # LIS length = length of first row
    # LDS length = length of first column (number of rows)
    lis_length = len(p[0]) if p else 0
    lds_length = len(p)

    return RSKResult(
        p_tableau=tuple(tuple(row) for row in p),
        q_tableau=tuple(tuple(row) for row in q),
        shape=shape,
        lis_length=lis_length,
        lds_length=lds_length,
    )
