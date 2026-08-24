"""Tests for primorial result-contract consistency (#2049)."""

from __future__ import annotations

import pytest

from jacobian.math.number_theory._models import (
    PositiveIntegerRequest,
    PrimorialResult,
)
from jacobian.math.number_theory._prime_operations import compute_primorial
from jacobian.math.number_theory._primes import PRIME_OPERATIONS


def test_primorial_boundary_113() -> None:
    """n=113 returns exactly 256 digits (the old BoundedInteger limit)."""
    result = compute_primorial(PositiveIntegerRequest(n=113))
    assert isinstance(result, PrimorialResult)
    assert len(result.value) == 256


def test_primorial_boundary_114() -> None:
    """n=114 returns 259 digits, exceeding the old BoundedInteger limit."""
    result = compute_primorial(PositiveIntegerRequest(n=114))
    assert isinstance(result, PrimorialResult)
    assert len(result.value) == 259


def test_primorial_maximum_1000() -> None:
    """The maximum accepted n returns a valid declared result."""
    result = compute_primorial(PositiveIntegerRequest(n=1000))
    assert isinstance(result, PrimorialResult)
    assert len(result.value) == 3393


def test_primorial_rejects_above_1000() -> None:
    """n=10001 is rejected by the request model before backend work."""
    with pytest.raises(ValueError):
        PositiveIntegerRequest(n=10001)


def test_primorial_5() -> None:
    """Primorial(5) = 2*3*5*7*11 = 2310."""
    result = compute_primorial(PositiveIntegerRequest(n=5))
    assert result.value == "2310"


def test_primorial_contract_version_tracks_the_result_schema_change() -> None:
    operation = next(
        item
        for item in PRIME_OPERATIONS
        if item.operation_id == "integer.compute.primorial"
    )
    assert operation.version == "3"
