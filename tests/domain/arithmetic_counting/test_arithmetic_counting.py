"""Tests for arithmetic counting operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.arithmetic_counting import (
    CongruenceConstrainedCountRequest,
    FloorSumRequest,
)
from jacobian.domains.arithmetic_counting.operations import (
    compute_congruence_constrained_count,
    compute_floor_sum,
)


def test_floor_sum_simple():
    """sum_{i=0}^{3} floor((2*i+1)/3) = 0+1+1+2 = 4."""
    result = compute_floor_sum(
        FloorSumRequest.model_validate({"n": 4, "m": 3, "a": 2, "b": 1})
    )
    assert result.value == 4


def test_floor_sum_zero():
    """sum with n=0 should be 0."""
    result = compute_floor_sum(
        FloorSumRequest.model_validate({"n": 0, "m": 1, "a": 0, "b": 0})
    )
    assert result.value == 0


def test_floor_sum_all_zero():
    """All a=0, b=0 means all floors are 0."""
    result = compute_floor_sum(
        FloorSumRequest.model_validate({"n": 10, "m": 5, "a": 0, "b": 0})
    )
    assert result.value == 0


def test_congruence_count_simple():
    """For k=3, m=2, n=1, lower=1, upper=2:
    b1=1: b2=1(1+1>=3? no), b2=2(1+2>=3? yes, 2≡1*1 mod 3? 2≠1, no) -> 0
    b1=2: b2=1(2+1>=3? yes, 1≡1*2 mod 3? 1≠2, no), b2=2(2+2>=3? yes, 2≡2 mod 3? yes) -> 1
    Total: 1
    """
    result = compute_congruence_constrained_count(
        CongruenceConstrainedCountRequest.model_validate({
            "k": 3, "m": 2, "n": 1, "lower": 1, "upper": 2,
        })
    )
    assert result.count == 1


def test_congruence_count_no_constraints():
    """Lower > upper should fail."""
    with pytest.raises(ValidationError, match="lower must be"):
        CongruenceConstrainedCountRequest.model_validate({
            "k": 3, "m": 2, "n": 1, "lower": 5, "upper": 1,
        })


def test_operations_discoverable():
    """Both operations should be discoverable."""
    from jacobian.domains.arithmetic_counting import arithmetic_counting_operations

    ops = arithmetic_counting_operations()
    op_ids = [op.operation_id for op in ops]
    assert "arithmetic.floor_sum.compute" in op_ids
    assert "arithmetic.congruence_constrained.count" in op_ids
