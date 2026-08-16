"""MathTool declarations for constraint-satisfaction construction."""

from __future__ import annotations

from jacobian.contracts.candidate_construction import (
    IntegerFeasibilityCheckRequest,
    IntegerFeasibilityCheckResult,
    IntegerFeasibilityRequest,
    IntegerFeasibilityResult,
)
from jacobian.domains._examples import example
from jacobian.domains.candidate_construction.operations import (
    construct_integer_feasibility,
    verify_integer_feasibility,
)
from jacobian.math_tools import MathTool


CANDIDATE_CONSTRUCTION_OPERATIONS: tuple[MathTool, ...] = (
    MathTool(
        operation_id="candidate.construct.integer_feasibility",
        version="1",
        title="Construct one feasible integer point",
        description=(
            "Use Z3 SMT to find one feasible integer point satisfying a set "
            "of declared linear constraints, or return a bounded infeasible "
            "result."
        ),
        request_type=IntegerFeasibilityRequest,
        result_type=IntegerFeasibilityResult,
        run=construct_integer_feasibility,
        tags=(
            "optimization",
            "integer",
            "feasibility",
            "z3",
            "smt",
            "bounded",
            "construction",
        ),
        examples=(
            example(
                "simple_feasible",
                "Find a feasible point for x + y <= 5, x >= 0, y >= 0.",
                {
                    "variable_count": 2,
                    "constraints": [
                        {"coefficients": [1, 1], "rhs": 5, "relation": "LE"},
                        {"coefficients": [-1, 0], "rhs": 0, "relation": "GE"},
                        {"coefficients": [0, -1], "rhs": 0, "relation": "GE"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="candidate.verify.integer_feasibility",
        version="1",
        title="Independently verify an integer feasibility assignment",
        description=(
            "Independently evaluate an integer assignment against a set of "
            "declared linear constraints and return whether it satisfies all."
        ),
        request_type=IntegerFeasibilityCheckRequest,
        result_type=IntegerFeasibilityCheckResult,
        run=verify_integer_feasibility,
        tags=(
            "optimization",
            "integer",
            "feasibility",
            "verify",
            "predicate",
        ),
        examples=(
            example(
                "simple_check",
                "Check that x=2, y=1 satisfies x + y <= 5.",
                {
                    "variable_count": 2,
                    "constraints": [
                        {"coefficients": [1, 1], "rhs": 5, "relation": "LE"},
                    ],
                    "assignment": [2, 1],
                },
            ),
        ),
    ),
)

__all__ = ["CANDIDATE_CONSTRUCTION_OPERATIONS"]
