from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.r_full_enumeration._models import (
    MAX_R_FULL_SIEVE_BOUND,
)
from jacobian.math.number_theory.r_full_enumeration.operations import (
    enumerate_r_full,
)


def test_powerful_small() -> None:
    """2-full (powerful) numbers up to 50."""
    result = enumerate_r_full(50, 2)
    # Powerful numbers: 1, 4, 8, 9, 16, 25, 27, 32, 36, 49
    assert 1 in result.values
    assert 4 in result.values
    assert 8 in result.values
    assert 9 in result.values
    assert 16 in result.values
    assert 25 in result.values
    assert 27 in result.values
    assert 32 in result.values
    assert 36 in result.values
    assert 49 in result.values
    # 2 is not powerful
    assert 2 not in result.values
    # 12 = 2^2 * 3 is not powerful (3 has exponent 1)
    assert 12 not in result.values


def test_cubefull_small() -> None:
    """3-full (cubefull) numbers up to 100."""
    result = enumerate_r_full(100, 3)
    # 1, 8, 16, 27, 32, 64, 81
    assert 1 in result.values
    assert 8 in result.values
    assert 27 in result.values
    assert 64 in result.values
    assert 81 in result.values
    # 4 is not cubefull (2^2, exponent 2 < 3)
    assert 4 not in result.values
    # 9 is not cubefull
    assert 9 not in result.values


def test_sorted() -> None:
    result = enumerate_r_full(100, 2)
    assert list(result.values) == sorted(result.values)


def test_count() -> None:
    result = enumerate_r_full(100, 2)
    assert result.count == len(result.values)


def test_empty() -> None:
    result = enumerate_r_full(0, 2)
    assert result.count == 0


def test_sieve_allocation_is_bounded_before_construction() -> None:
    with pytest.raises(OperationDomainValidationError, match="0 through"):
        enumerate_r_full(MAX_R_FULL_SIEVE_BOUND + 1, 2)


def test_invalid_minimum_exponent_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="at least 2"):
        enumerate_r_full(10, 1)


def test_huge_exponent_is_bounded_without_constructing_the_power() -> None:
    result = enumerate_r_full(2, 1_000_000_000)
    assert result.values == (1,)
