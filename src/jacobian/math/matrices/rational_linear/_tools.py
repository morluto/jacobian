"""Exact rational-linear operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.matrices.rational_linear._models import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalInconsistencyResult,
    LinearRationalSolutionFindRequest,
    LinearRationalSolutionResult,
)
from jacobian.math.matrices.rational_linear.operations import (
    inconsistency_witness,
    solve,
)


def compute_rational_solution(
    request: LinearRationalSolutionFindRequest,
) -> LinearRationalSolutionResult:
    values = solve(request.system)
    if values is None:
        return LinearRationalSolutionResult._from_kernel(
            system=request.system, status="INCONSISTENT"
        )
    return LinearRationalSolutionResult._from_kernel(
        system=request.system, status="SOLUTION", values=values
    )


def compute_rational_inconsistency(
    request: LinearRationalInconsistencyFindRequest,
) -> LinearRationalInconsistencyResult:
    witness = inconsistency_witness(request.system)
    if witness is None:
        return LinearRationalInconsistencyResult._from_kernel(
            system=request.system, status="CONSISTENT"
        )
    left_witness, rhs_pairing = witness
    return LinearRationalInconsistencyResult._from_kernel(
        system=request.system,
        status="INCONSISTENT",
        left_witness=left_witness,
        rhs_pairing=rhs_pairing,
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="linear.rational_solution.compute",
        title="Compute an exact rational solution",
        description="Return an exact rational solution or an inconsistent outcome for one canonical coordinate-sparse system, subject to nonzero, scalar-work, and result-height bounds.",
        request_type=LinearRationalSolutionFindRequest,
        result_type=LinearRationalSolutionResult,
        run=compute_rational_solution,
        tags=("linear-algebra", "rational", "solution", "exact"),
        examples=(
            OperationExample(
                name="identity_solution",
                description="Solve a one-variable identity system.",
                input={
                    "system": {
                        "variables": ["x"],
                        "coefficients": {
                            "row_count": 1,
                            "column_count": 1,
                            "entries": [
                                {
                                    "row": 0,
                                    "column": 0,
                                    "value": {"num": "1", "den": "1"},
                                }
                            ],
                        },
                        "rhs": [{"num": "2", "den": "1"}],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="linear.rational_inconsistency.compute",
        title="Compute an exact rational inconsistency witness",
        description="Return an exact left inconsistency witness or a consistent outcome for one canonical coordinate-sparse system, subject to nonzero, scalar-work, and result-height bounds.",
        request_type=LinearRationalInconsistencyFindRequest,
        result_type=LinearRationalInconsistencyResult,
        run=compute_rational_inconsistency,
        tags=("linear-algebra", "rational", "inconsistency", "exact"),
        examples=(
            OperationExample(
                name="contradictory_one_variable_system",
                description="Find a witness for x=0 together with x=1.",
                input={
                    "system": {
                        "variables": ["x"],
                        "coefficients": {
                            "row_count": 2,
                            "column_count": 1,
                            "entries": [
                                {
                                    "row": 0,
                                    "column": 0,
                                    "value": {"num": "1", "den": "1"},
                                },
                                {
                                    "row": 1,
                                    "column": 0,
                                    "value": {"num": "1", "den": "1"},
                                },
                            ],
                        },
                        "rhs": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
