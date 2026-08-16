"""MathTool declarations for finite-instance testing."""

from __future__ import annotations

from jacobian.contracts.finite_instance_testing import (
    FiniteInstanceTestRequest,
    FiniteInstanceTestResult,
)
from jacobian.domains._examples import example
from jacobian.domains.finite_instance_testing.operations import (
    compute_finite_instance_test,
)
from jacobian.math_tools import MathTool


FINITE_INSTANCE_TESTING_OPERATIONS: tuple[MathTool, ...] = (
    MathTool(
        operation_id="claim.test.instances",
        version="1",
        title="Test a quantified claim over a finite instance set",
        description=(
            "Evaluate a quantified claim on each instance in an explicit "
            "finite set and return per-instance results with exact coverage "
            "accounting."
        ),
        request_type=FiniteInstanceTestRequest,
        result_type=FiniteInstanceTestResult,
        run=compute_finite_instance_test,
        tags=(
            "testing",
            "finite",
            "instance",
            "claim",
            "bounded",
            "coverage",
        ),
        examples=(
            example(
                "even_numbers",
                "Test that all numbers in a set are even.",
                {
                    "claim_name": "even",
                    "instances": [
                        {"key": "two", "payload": "2"},
                        {"key": "four", "payload": "4"},
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["FINITE_INSTANCE_TESTING_OPERATIONS"]
