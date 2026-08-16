"""Tests for bounded constraint-satisfaction construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.candidate_construction import IntegerFeasibilityRequest
from jacobian.domains.candidate_construction.operations import (
    construct_integer_feasibility,
    verify_integer_feasibility,
)


def test_satisfiable_constraint():
    """x + y <= 5, x >= 0, y >= 0 should be feasible."""
    request = IntegerFeasibilityRequest.model_validate({
        "variable_count": 2,
        "constraints": [
            {"coefficients": [1, 1], "rhs": 5, "relation": "LE"},
            {"coefficients": [-1, 0], "rhs": 0, "relation": "GE"},
            {"coefficients": [0, -1], "rhs": 0, "relation": "GE"},
        ],
    })
    result = construct_integer_feasibility(request)
    assert result.status == "FEASIBLE"
    assert result.assignment is not None
    assert len(result.assignment) == 2


def test_unsatisfiable_constraint():
    """x <= 0 and x >= 1 is infeasible."""
    request = IntegerFeasibilityRequest.model_validate({
        "variable_count": 1,
        "constraints": [
            {"coefficients": [1], "rhs": 0, "relation": "LE"},
            {"coefficients": [1], "rhs": 1, "relation": "GE"},
        ],
    })
    result = construct_integer_feasibility(request)
    assert result.status == "INFEASIBLE"


def test_verify_satisfying_assignment():
    """Verify that x=2, y=1 satisfies x + y <= 5."""
    from jacobian.contracts.candidate_construction import IntegerFeasibilityCheckRequest

    request = IntegerFeasibilityCheckRequest.model_validate({
        "variable_count": 2,
        "constraints": [
            {"coefficients": [1, 1], "rhs": 5, "relation": "LE"},
        ],
        "assignment": [2, 1],
    })
    result = verify_integer_feasibility(request)
    assert result.satisfies is True


def test_verify_violating_assignment():
    """Verify that x=3, y=3 does NOT satisfy x + y <= 5."""
    from jacobian.contracts.candidate_construction import IntegerFeasibilityCheckRequest

    request = IntegerFeasibilityCheckRequest.model_validate({
        "variable_count": 2,
        "constraints": [
            {"coefficients": [1, 1], "rhs": 5, "relation": "LE"},
        ],
        "assignment": [3, 3],
    })
    result = verify_integer_feasibility(request)
    assert result.satisfies is False
    assert result.first_violated_constraint == 0


def test_constraint_dimension_mismatch():
    """A constraint with wrong coefficient count should fail."""
    with pytest.raises(ValidationError, match="variable_count"):
        IntegerFeasibilityRequest.model_validate({
            "variable_count": 2,
            "constraints": [
                {"coefficients": [1, 2, 3], "rhs": 5, "relation": "LE"},
            ],
        })


def test_operations_discoverable():
    """Both operations should be discoverable via the factory."""
    from jacobian.domains.candidate_construction import (
        candidate_construction_operations,
    )

    ops = candidate_construction_operations()
    op_ids = [op.operation_id for op in ops]
    assert "candidate.construct.integer_feasibility" in op_ids
    assert "candidate.verify.integer_feasibility" in op_ids
