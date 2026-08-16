"""Tests for exact optimality verification."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.optimality_verification import (
    RationalOptimalityVerifyRequest,
)
from jacobian.domains.optimality_verification.operations import (
    verify_rational_optimality,
)


def _make_lp():
    """Make a simple LP: min(x) s.t. x = 3, x >= 0."""
    return {
        "variables": ["x"],
        "objective": [{"num": "1", "den": "1"}],
        "coefficients": [[{"num": "1", "den": "1"}]],
        "rhs": [{"num": "3", "den": "1"}],
    }


def test_correct_optimum_is_verified():
    """A correct primal and dual should be verified."""
    request = RationalOptimalityVerifyRequest.model_validate({
        "program": _make_lp(),
        "claimed_objective": {"num": "3", "den": "1"},
        "primal_candidate": [{"num": "3", "den": "1"}],
        "dual_candidate": [{"num": "1", "den": "1"}],
    })
    result = verify_rational_optimality(request)
    assert result.status == "VERIFIED"


def test_primal_corruption_rejected():
    """A primal that violates constraints should be rejected."""
    request = RationalOptimalityVerifyRequest.model_validate({
        "program": _make_lp(),
        "claimed_objective": {"num": "3", "den": "1"},
        "primal_candidate": [{"num": "5", "den": "1"}],
        "dual_candidate": [{"num": "1", "den": "1"}],
    })
    result = verify_rational_optimality(request)
    assert result.status == "REJECTED"


def test_dual_corruption_rejected():
    """A dual that violates the dual constraint should be rejected."""
    request = RationalOptimalityVerifyRequest.model_validate({
        "program": _make_lp(),
        "claimed_objective": {"num": "3", "den": "1"},
        "primal_candidate": [{"num": "3", "den": "1"}],
        "dual_candidate": [{"num": "5", "den": "1"}],
    })
    result = verify_rational_optimality(request)
    assert result.status == "REJECTED"


def test_claimed_objective_mismatch_rejected():
    """A wrong claimed objective should be rejected."""
    request = RationalOptimalityVerifyRequest.model_validate({
        "program": _make_lp(),
        "claimed_objective": {"num": "4", "den": "1"},
        "primal_candidate": [{"num": "3", "den": "1"}],
        "dual_candidate": [{"num": "1", "den": "1"}],
    })
    result = verify_rational_optimality(request)
    assert result.status == "REJECTED"


def test_dimension_mismatch_rejected():
    """Mismatched primal/dual dimensions should fail validation."""
    with pytest.raises(ValidationError, match="primal candidate"):
        RationalOptimalityVerifyRequest.model_validate({
            "program": _make_lp(),
            "claimed_objective": {"num": "3", "den": "1"},
            "primal_candidate": [{"num": "3", "den": "1"}, {"num": "0", "den": "1"}],
            "dual_candidate": [{"num": "1", "den": "1"}],
        })


def test_operation_discoverable():
    """The operation should be discoverable via the factory."""
    from jacobian.domains.optimality_verification import (
        optimality_verification_operations,
    )

    ops = optimality_verification_operations()
    assert any(op.operation_id == "optimality.verify.rational_lp" for op in ops)
