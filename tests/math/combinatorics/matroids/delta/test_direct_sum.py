"""Exact disjoint-ground direct sums of finite delta-matroids."""

from __future__ import annotations

import pytest

from jacobian.math.combinatorics.matroids.delta._tools import _run_direct_sum
from jacobian.math.combinatorics.matroids.delta.extra import (
    DeltaMatroidDirectSumRequest,
)
from jacobian.math.combinatorics.matroids.delta.operations import direct_sum
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid


def test_direct_sum_matches_independent_pairwise_union_oracle() -> None:
    left = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1), (1,)))
    right = FiniteDeltaMatroid(ground=("c",), feasible=((), (0,)))

    result = direct_sum(left, right)

    expected = tuple(
        sorted(
            tuple(sorted((*left_row, *(len(left.ground) + i for i in right_row))))
            for left_row in left.feasible
            for right_row in right.feasible
        )
    )
    assert result.direct_sum.ground == ("a", "b", "c")
    assert result.direct_sum.feasible == expected
    assert result.left_injection == (0, 1)
    assert result.right_injection == (2,)
    # The product family is itself a delta-matroid (here the complete cube).
    assert len(result.direct_sum.feasible) == 8


def test_direct_sum_empty_ground_is_identity_on_feasible_rows() -> None:
    identity = FiniteDeltaMatroid(ground=(), feasible=((),))
    value = FiniteDeltaMatroid(ground=("x",), feasible=((), (0,)))
    result = direct_sum(identity, value)
    assert result.direct_sum == value
    assert result.left_injection == ()
    assert result.right_injection == (0,)


def test_overlapping_labels_rejected_instead_of_silently_tagged() -> None:
    left = FiniteDeltaMatroid(ground=("x",), feasible=((), (0,)))
    right = FiniteDeltaMatroid(ground=("x",), feasible=((), (0,)))
    with pytest.raises(ValueError, match="disjoint"):
        _run_direct_sum(DeltaMatroidDirectSumRequest(left=left, right=right))


def test_catalog_path_returns_typed_result() -> None:
    left = FiniteDeltaMatroid(ground=("a",), feasible=((), (0,)))
    right = FiniteDeltaMatroid(ground=("b",), feasible=((),))
    result = _run_direct_sum(DeltaMatroidDirectSumRequest(left=left, right=right))
    assert result.direct_sum.feasible == ((), (0,))
