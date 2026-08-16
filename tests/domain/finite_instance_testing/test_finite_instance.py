"""Tests for bounded finite-instance claim testing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.finite_instance_testing import FiniteInstanceTestRequest
from jacobian.domains.finite_instance_testing.operations import (
    compute_finite_instance_test,
)


def test_universal_claim_holds():
    """A universal claim that holds on all instances should return COMPUTED."""
    request = FiniteInstanceTestRequest.model_validate({
        "claim_name": "even",
        "instances": [
            {"key": "two", "payload": "2"},
            {"key": "four", "payload": "4"},
            {"key": "six", "payload": "6"},
        ],
    })
    result = compute_finite_instance_test(request)
    assert result.status == "COMPUTED"
    assert result.instance_count == 3
    assert result.passed_count == 3


def test_universal_claim_violated():
    """A universal claim that fails on one instance should return VIOLATED."""
    request = FiniteInstanceTestRequest.model_validate({
        "claim_name": "even",
        "instances": [
            {"key": "two", "payload": "2"},
            {"key": "three", "payload": "3"},
            {"key": "four", "payload": "4"},
        ],
    })
    result = compute_finite_instance_test(request)
    assert result.status == "VIOLATED"
    assert result.instance_count == 3
    assert result.passed_count == 2


def test_empty_instance_set():
    """An empty instance set should return EMPTY."""
    request = FiniteInstanceTestRequest.model_validate({
        "claim_name": "even",
        "instances": [],
    })
    result = compute_finite_instance_test(request)
    assert result.status == "EMPTY"
    assert result.instance_count == 0


def test_prime_claim():
    """Test the prime claim."""
    request = FiniteInstanceTestRequest.model_validate({
        "claim_name": "prime",
        "instances": [
            {"key": "two", "payload": "2"},
            {"key": "three", "payload": "3"},
            {"key": "five", "payload": "5"},
        ],
    })
    result = compute_finite_instance_test(request)
    assert result.status == "COMPUTED"
    assert result.passed_count == 3


def test_positive_claim_violated():
    """Test the positive claim with a negative number."""
    request = FiniteInstanceTestRequest.model_validate({
        "claim_name": "positive",
        "instances": [
            {"key": "one", "payload": "1"},
            {"key": "neg", "payload": "-5"},
        ],
    })
    result = compute_finite_instance_test(request)
    assert result.status == "VIOLATED"
    assert result.passed_count == 1


def test_duplicate_keys_rejected():
    """Duplicate instance keys should fail validation."""
    with pytest.raises(ValidationError, match="instance keys must be unique"):
        FiniteInstanceTestRequest.model_validate({
            "claim_name": "even",
            "instances": [
                {"key": "a", "payload": "2"},
                {"key": "a", "payload": "4"},
            ],
        })


def test_operation_discoverable():
    """The operation should be discoverable via the factory."""
    from jacobian.domains.finite_instance_testing import (
        finite_instance_testing_operations,
    )

    ops = finite_instance_testing_operations()
    assert any(op.operation_id == "claim.test.instances" for op in ops)
