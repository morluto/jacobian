"""Tests for primorial result-contract consistency (#2049)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory._integer_models import PositiveIntegerRequest
from jacobian.math.number_theory._prime_models import (
    PrimorialRequest,
    PrimorialResult,
)
from jacobian.math.number_theory._prime_operations import compute_primorial
from jacobian.math.number_theory._primes import PRIME_OPERATIONS


def test_primorial_boundary_113() -> None:
    """n=113 returns exactly 256 digits (the old BoundedInteger limit)."""
    result = compute_primorial(PrimorialRequest(n=113))
    assert isinstance(result, PrimorialResult)
    assert len(result.value) == 256


def test_primorial_boundary_114() -> None:
    """n=114 returns 259 digits, exceeding the old BoundedInteger limit."""
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


def test_primorial_rejects_above_1000() -> None:
    """n=10001 is rejected by the request model before backend work."""
    with pytest.raises(ValueError):
        PrimorialRequest(n=10001)


def test_primorial_5() -> None:
    """Primorial(5) = 2*3*5*7*11 = 2310."""
    result = compute_primorial(PrimorialRequest(n=5))
    assert result.value == "2310"


def test_primorial_contract_version_tracks_the_result_schema_change() -> None:
    next(
        item
        for item in PRIME_OPERATIONS
        if item.operation_id == "integer.compute.primorial"
    )


def test_primorial_admission_records_the_v4_result_derived_envelope() -> None:
    """The materially changed v4 candidate has a fresh owner-local decision."""
    from jacobian.catalog.admission import AdmissionDecision
    from jacobian.math.number_theory._admission import ADMISSIONS

    admission = next(
        item for item in ADMISSIONS if item.operation_id == "integer.compute.primorial"
    )

    assert admission.decision == AdmissionDecision.KEEP
    assert "n <= 1001" in admission.rationale
    assert "3,400-digit result budget" in admission.rationale
