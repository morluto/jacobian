"""Tests for exact partition and tableau operations added for #1825."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.algebraic._models import (
    HookContentCountRequest,
    PartitionDominanceRequest,
    SemistandardTableauCheckRequest,
    StandardTableauCheckRequest,
)
from jacobian.math.combinatorics.algebraic._tools import (
    check_semistandard_tableau,
    check_standard_tableau,
    hook_content_count,
    partition_dominance,
)


def test_hook_content_count_and_factors() -> None:
    result = hook_content_count(
        HookContentCountRequest(partition={"parts": [2, 1]}, alphabet_size=2)
    )
    assert result.count == 2
    assert result.numerators == (2, 3, 1)
    assert result.hook_product == 3


def test_hook_content_zero_when_alphabet_is_too_small() -> None:
    result = hook_content_count(
        HookContentCountRequest(partition={"parts": [1, 1]}, alphabet_size=1)
    )
    assert result.count == 0


def test_hook_content_empty_shape() -> None:
    result = hook_content_count(
        HookContentCountRequest(partition={"parts": []}, alphabet_size=3)
    )
    assert result.count == 1
    assert result.numerators == ()
    assert result.hook_product == 1


@pytest.mark.parametrize(
    ("left", "right", "relation"),
    [
        ((2, 1), (1, 1, 1), "LEFT_DOMINATES"),
        ((1, 1, 1), (2, 1), "RIGHT_DOMINATES"),
        ((2, 2), (2, 2), "EQUAL"),
        ((3, 1, 1, 1), (2, 2, 2), "INCOMPARABLE"),
    ],
)
def test_partition_dominance_ledger(
    left: tuple[int, ...], right: tuple[int, ...], relation: str
) -> None:
    result = partition_dominance(
        PartitionDominanceRequest(left={"parts": left}, right={"parts": right})
    )
    assert result.relation == relation
    assert len(result.left_prefix_sums) == max(len(left), len(right))
    assert result.left_prefix_sums[-1] == result.right_prefix_sums[-1]


def test_partition_dominance_different_sizes_is_distinct() -> None:
    result = partition_dominance(
        PartitionDominanceRequest(left={"parts": [2]}, right={"parts": [1]})
    )
    assert result.relation == "NOT_COMPARABLE_DIFFERENT_SIZE"
    assert result.left_prefix_sums == ()


def test_tableau_checkers_replay_membership() -> None:
    standard = check_standard_tableau(
        StandardTableauCheckRequest(tableau={"rows": [[1, 2], [3]]})
    )
    semistandard = check_semistandard_tableau(
        SemistandardTableauCheckRequest(tableau={"rows": [[1, 1], [2]]})
    )
    assert standard.rows == ((1, 2), (3,))
    assert semistandard.rows == ((1, 1), (2,))


@pytest.mark.parametrize(
    "candidate",
    [
        StandardTableauCheckRequest(tableau={"rows": [[1, 1], [2]]}),
        StandardTableauCheckRequest(tableau={"rows": [[1, 2], [2]]}),
    ],
)
def test_standard_tableau_checker_rejects_row_or_column_failure(candidate) -> None:
    with pytest.raises(ValueError):
        check_standard_tableau(candidate)


def test_semistandard_tableau_checker_rejects_column_failure() -> None:
    request = SemistandardTableauCheckRequest(tableau={"rows": [[1, 2], [1]]})
    with pytest.raises(ValueError):
        check_semistandard_tableau(request)


def test_hook_content_alphabet_bound_is_published() -> None:
    with pytest.raises(ValidationError):
        HookContentCountRequest(partition={"parts": [1]}, alphabet_size=0)
