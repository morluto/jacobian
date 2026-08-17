from __future__ import annotations

from itertools import permutations

import pytest
from pydantic import ValidationError

from jacobian.math.algebraic_combinatorics._models import (
    ConjugatePartitionRequest,
    HookLengthRequest,
    Partition,
    StandardYoungTableauCountRequest,
)
from jacobian.math.algebraic_combinatorics._operations import (
    compute_conjugate_partition,
    compute_hook_lengths,
    compute_syt_count,
)


def test_hook_lengths_partition_321() -> None:
    """Hook lengths of (3,2,1) are [[5,3,1],[3,1],[1]]."""
    result = compute_hook_lengths(
        HookLengthRequest(partition=Partition(parts=(3, 2, 1)))
    )
    assert result.hooks == ((5, 3, 1), (3, 1), (1,))
    assert result.total_product == "45"


def test_hook_lengths_single_row() -> None:
    """Hook lengths of (n) are [n, n-1, ..., 1]."""
    result = compute_hook_lengths(HookLengthRequest(partition=Partition(parts=(4,))))
    assert result.hooks == ((4, 3, 2, 1),)
    assert result.total_product == "24"


def test_hook_lengths_single_column() -> None:
    """Hook lengths of (1,1,1) are [[1],[1],[1]]."""
    result = compute_hook_lengths(
        HookLengthRequest(partition=Partition(parts=(1, 1, 1)))
    )
    assert result.hooks == ((3,), (2,), (1,))
    assert result.total_product == "6"


def test_syt_count_partition_321() -> None:
    """Number of SYT for shape (3,2,1) is 16."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=Partition(parts=(3, 2, 1)))
    )
    assert result.count == "16"
    assert result.n == 6
    assert result.method == "HOOK_LENGTH_FORMULA"


def test_syt_count_single_row() -> None:
    """Number of SYT for shape (n) is 1."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=Partition(parts=(5,)))
    )
    assert result.count == "1"
    assert result.n == 5


def test_syt_count_single_column() -> None:
    """Number of SYT for shape (1,1,...,1) is 1."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=Partition(parts=(1, 1, 1, 1)))
    )
    assert result.count == "1"
    assert result.n == 4


def test_syt_count_rectangle_22() -> None:
    """Number of SYT for shape (2,2) is 2."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=Partition(parts=(2, 2)))
    )
    assert result.count == "2"


def _count_syt_brute_force(parts: tuple[int, ...]) -> int:
    """Brute-force count of standard Young tableaux."""
    n = sum(parts)
    positions = []
    for row, length in enumerate(parts):
        for column in range(length):
            positions.append((row, column))

    def is_valid(perm: tuple[int, ...]) -> bool:
        table = {positions[idx]: perm[idx] for idx in range(len(perm))}
        for row, column in positions:
            if row > 0 and table[(row, column)] <= table[(row - 1, column)]:
                return False
            if column > 0 and table[(row, column)] <= table[(row, column - 1)]:
                return False
        return True

    return sum(1 for perm in permutations(range(1, n + 1)) if is_valid(perm))


def test_syt_count_matches_brute_force() -> None:
    """SYT count matches the number of valid fillings for small partitions."""
    for parts in [(3, 1), (2, 2), (3, 2)]:
        brute = _count_syt_brute_force(parts)
        result = compute_syt_count(
            StandardYoungTableauCountRequest(partition=Partition(parts=parts))
        )
        assert result.count == str(brute)


def test_conjugate_self_conjugate_partition() -> None:
    """Conjugate of (3,2,1) is (3,2,1) — self-conjugate."""
    result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=Partition(parts=(3, 2, 1)))
    )
    assert result.conjugate == (3, 2, 1)


def test_conjugate_row_to_column() -> None:
    """Conjugate of (4) is (1,1,1,1) and vice versa."""
    result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=Partition(parts=(4,)))
    )
    assert result.conjugate == (1, 1, 1, 1)


def test_conjugate_column_to_row() -> None:
    """Conjugate of (1,1,1,1) is (4)."""
    result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=Partition(parts=(1, 1, 1, 1)))
    )
    assert result.conjugate == (4,)


def test_conjugate_double_conjugate_is_identity() -> None:
    """Conjugate of conjugate is the original partition."""
    result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=Partition(parts=(5, 3, 2, 1)))
    )
    result2 = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=Partition(parts=result.conjugate))
    )
    assert result2.conjugate == (5, 3, 2, 1)


def test_contract_rejects_non_decreasing() -> None:
    with pytest.raises(ValidationError, match="non-increasing"):
        Partition(parts=(1, 2, 3))


def test_contract_rejects_non_positive() -> None:
    with pytest.raises(ValidationError, match="positive"):
        Partition(parts=(3, 0, 1))


def test_contract_rejects_partition_exceeding_size_bound() -> None:
    """A single-part partition summing above MAX_PARTITION_SIZE is rejected."""
    with pytest.raises(ValidationError, match="partition size must not exceed"):
        Partition(parts=(51,))


def test_contract_rejects_non_integer_parts() -> None:
    """Boolean or string partition parts are rejected, not silently coerced."""
    with pytest.raises(ValidationError):
        Partition.model_validate({"parts": [True]})
    with pytest.raises(ValidationError):
        Partition.model_validate({"parts": ["3"]})


def test_syt_count_large_returns_canonical_string() -> None:
    """Large SYT counts are returned as canonical decimal strings."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(
            partition=Partition(parts=(10, 9, 8, 7, 6, 5, 4, 1))
        )
    )
    assert result.count == "322821557622027077916662169600"
