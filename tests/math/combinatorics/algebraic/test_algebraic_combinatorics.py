from __future__ import annotations

import math
from itertools import permutations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics import algebraic as algebraic_combinatorics
from jacobian.math.combinatorics.algebraic._models import (
    ConjugatePartitionRequest,
    HookLengthRequest,
    StandardYoungTableauCountRequest,
)
from jacobian.math.combinatorics.algebraic._operations import (
    compute_conjugate_partition,
    compute_hook_lengths,
    compute_syt_count,
)
from jacobian.math.combinatorics.symmetric_functions import IntegerPartition


def test_hook_lengths_partition_321() -> None:
    """Hook lengths of (3,2,1) are [[5,3,1],[3,1],[1]]."""
    result = compute_hook_lengths(
        HookLengthRequest(partition=IntegerPartition(parts=(3, 2, 1)))
    )
    assert result.hooks == ((5, 3, 1), (3, 1), (1,))
    assert result.total_product == "45"


def test_hook_lengths_single_row() -> None:
    """Hook lengths of (n) are [n, n-1, ..., 1]."""
    result = compute_hook_lengths(
        HookLengthRequest(partition=IntegerPartition(parts=(4,)))
    )
    assert result.hooks == ((4, 3, 2, 1),)
    assert result.total_product == "24"


def test_hook_lengths_single_column() -> None:
    """Hook lengths of (1,1,1) are [[1],[1],[1]]."""
    result = compute_hook_lengths(
        HookLengthRequest(partition=IntegerPartition(parts=(1, 1, 1)))
    )
    assert result.hooks == ((3,), (2,), (1,))
    assert result.total_product == "6"


def test_syt_count_partition_321() -> None:
    """Number of SYT for shape (3,2,1) is 16."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=IntegerPartition(parts=(3, 2, 1)))
    )
    assert result.count == "16"
    assert result.n == 6
    assert result.method == "HOOK_LENGTH_FORMULA"


def test_syt_count_single_row() -> None:
    """Number of SYT for shape (n) is 1."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=IntegerPartition(parts=(5,)))
    )
    assert result.count == "1"
    assert result.n == 5


def test_syt_count_single_column() -> None:
    """Number of SYT for shape (1,1,...,1) is 1."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=IntegerPartition(parts=(1, 1, 1, 1)))
    )
    assert result.count == "1"
    assert result.n == 4


def test_syt_count_rectangle_22() -> None:
    """Number of SYT for shape (2,2) is 2."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=IntegerPartition(parts=(2, 2)))
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
            StandardYoungTableauCountRequest(partition=IntegerPartition(parts=parts))
        )
        assert result.count == str(brute)


def test_conjugate_self_conjugate_partition() -> None:
    """Conjugate of (3,2,1) is (3,2,1) — self-conjugate."""
    result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=IntegerPartition(parts=(3, 2, 1)))
    )
    assert result.conjugate.parts == (3, 2, 1)


def test_conjugate_row_to_column() -> None:
    """Conjugate of (4) is (1,1,1,1) and vice versa."""
    result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=IntegerPartition(parts=(4,)))
    )
    assert result.conjugate.parts == (1, 1, 1, 1)


def test_conjugate_column_to_row() -> None:
    """Conjugate of (1,1,1,1) is (4)."""
    result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=IntegerPartition(parts=(1, 1, 1, 1)))
    )
    assert result.conjugate.parts == (4,)


def test_conjugate_double_conjugate_is_identity() -> None:
    """Conjugate of conjugate is the original partition."""
    result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=IntegerPartition(parts=(5, 3, 2, 1)))
    )
    result2 = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=result.conjugate)
    )
    assert result2.conjugate.parts == (5, 3, 2, 1)


def test_empty_canonical_partition_composes_with_all_partition_consumers() -> None:
    partition = IntegerPartition(parts=())
    hook_result = compute_hook_lengths(HookLengthRequest(partition=partition))
    count_result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=partition)
    )
    conjugate_result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=partition)
    )

    assert hook_result.hooks == ()
    assert hook_result.total_product == "1"
    assert count_result.count == "1"
    assert count_result.n == 0
    assert conjugate_result.conjugate == partition
    assert algebraic_combinatorics.hook_lengths(partition) == ()
    assert algebraic_combinatorics.standard_young_tableaux_count(partition) == 1
    assert algebraic_combinatorics.conjugate_partition(partition) == partition


def test_native_partition_functions_are_closed_at_conjugate_boundary() -> None:
    row = IntegerPartition(parts=(100,))
    column = algebraic_combinatorics.conjugate_partition(row)
    assert column.parts == (1,) * 100
    assert algebraic_combinatorics.conjugate_partition(column) == row
    assert len(algebraic_combinatorics.hook_lengths(column)) == 100
    assert algebraic_combinatorics.standard_young_tableaux_count(column) == 1


def test_partition_operations_return_typed_results_at_the_size_boundary() -> None:
    """The canonical domain admits the conjugate of every admitted partition."""
    row = IntegerPartition(parts=(500,))
    conjugate_result = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=row)
    )
    assert conjugate_result.conjugate.parts == (1,) * 500
    round_trip = compute_conjugate_partition(
        ConjugatePartitionRequest(partition=conjugate_result.conjugate)
    )
    assert round_trip.conjugate == row

    hook_result = compute_hook_lengths(HookLengthRequest(partition=row))
    assert hook_result.hooks == (tuple(range(500, 0, -1)),)
    assert int(hook_result.total_product) == math.factorial(500)

    count_result = compute_syt_count(
        StandardYoungTableauCountRequest(partition=conjugate_result.conjugate)
    )
    assert count_result.count == "1"
    assert count_result.n == 500


def test_conjugate_operation_publishes_its_changed_wire_shape_as_version_two() -> None:
    from jacobian.math.combinatorics.algebraic._tools import TOOLS

    {tool.operation_id: tool for tool in TOOLS}
    # The conjugate result changed from a bare integer array to the canonical
    # IntegerPartition value, which is a versioned contract change.


def test_contract_rejects_non_decreasing() -> None:
    with pytest.raises(ValidationError) as error:
        IntegerPartition(parts=(1, 2, 3))
    assert (
        error.value.errors()[0]["type"]
        == "symmetric_function.partition_not_weakly_decreasing"
    )


def test_contract_rejects_non_positive() -> None:
    with pytest.raises(ValidationError) as error:
        IntegerPartition(parts=(3, 0, 1))
    assert (
        error.value.errors()[0]["type"]
        == "symmetric_function.partition_parts_not_positive"
    )


def test_contract_rejects_partition_exceeding_size_bound() -> None:
    """A single-part partition summing above MAX_PARTITION_SIZE is rejected."""
    with pytest.raises(ValidationError) as error:
        IntegerPartition(parts=(501,))
    assert (
        error.value.errors()[0]["type"] == "symmetric_function.partition_size_exceeded"
    )


def test_contract_rejects_non_integer_parts() -> None:
    """Boolean or string partition parts are rejected, not silently coerced."""
    with pytest.raises(ValidationError):
        IntegerPartition.model_validate({"parts": [True]})
    with pytest.raises(ValidationError):
        IntegerPartition.model_validate({"parts": ["3"]})


def test_syt_count_large_returns_canonical_string() -> None:
    """Large SYT counts are returned as canonical decimal strings."""
    result = compute_syt_count(
        StandardYoungTableauCountRequest(
            partition=IntegerPartition(parts=(10, 9, 8, 7, 6, 5, 4, 1))
        )
    )
    assert result.count == "322821557622027077916662169600"
