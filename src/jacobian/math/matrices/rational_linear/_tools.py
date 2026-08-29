"""Exact rational-linear operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: MathTools = (
    _op(
        "linear.rational_solution.compute",
        "Compute an exact rational solution",
        "Return an exact rational solution or an inconsistent outcome for one canonical coordinate-sparse system, subject to nonzero, scalar-work, and result-height bounds.",
        LinearRationalSolutionFindRequest,
        LinearRationalSolutionResult,
        compute_rational_solution,
        "linear-algebra",
        "rational",
        "solution",
        "exact",
        examples=(
            example(
                "identity_solution",
                "Solve a one-variable identity system.",
                {
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
    _op(
        "linear.rational_inconsistency.compute",
        "Compute an exact rational inconsistency witness",
        "Return an exact left inconsistency witness or a consistent outcome for one canonical coordinate-sparse system, subject to nonzero, scalar-work, and result-height bounds.",
        LinearRationalInconsistencyFindRequest,
        LinearRationalInconsistencyResult,
        compute_rational_inconsistency,
        "linear-algebra",
        "rational",
        "inconsistency",
        "exact",
        examples=(
            example(
                "contradictory_one_variable_system",
                "Find a witness for x=0 together with x=1.",
                {
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
