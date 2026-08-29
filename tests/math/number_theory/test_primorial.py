"""Tests for the exact primorial result contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory._integer_models import PositiveIntegerRequest
from jacobian.math.number_theory._prime_models import (
    PrimorialRequest,
    PrimorialResult,
)
from jacobian.math.number_theory._primes import compute_primorial


def test_primorial_boundary_113() -> None:
    """n=113 returns exactly 256 digits."""
    result = compute_primorial(PrimorialRequest(n=113))
    assert isinstance(result, PrimorialResult)
    assert len(result.value) == 256


def test_primorial_boundary_114() -> None:
    """n=114 returns exactly 259 digits."""
    result = compute_primorial(PrimorialRequest(n=114))
    assert isinstance(result, PrimorialResult)
    assert len(result.value) == 259


def test_primorial_maximum_1000() -> None:
    """The maximum accepted n returns a valid declared result."""
    result = compute_primorial(PrimorialRequest(n=1000))
    assert isinstance(result, PrimorialResult)
    assert len(result.value) == 3393


def test_primorial_admits_exact_digit_boundary_1001() -> None:
    """primorial(1001) has 3397 digits and is admitted; 1002 (3401) is not."""
    result = compute_primorial(PrimorialRequest(n=1001))
    assert isinstance(result, PrimorialResult)
    assert len(result.value) == 3397
    with pytest.raises(ValidationError):
        PrimorialRequest(n=1002)


def test_positive_integer_request_still_covers_other_operations() -> None:
    """The shared arithmetic-function bound remains at 10,000."""
    PositiveIntegerRequest(n=10_000)


def test_primorial_5() -> None:
    """Primorial(5) = 2*3*5*7*11 = 2310."""
    result = compute_primorial(PrimorialRequest(n=5))
    assert result.value == "2310"
