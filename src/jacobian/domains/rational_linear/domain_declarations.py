"""Domain-owned rational-linear operation declarations."""

from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalInconsistencyResult,
    LinearRationalSolutionFindRequest,
    LinearRationalSolutionResult,
)
from jacobian.domains._examples import example
from jacobian.domains.rational_linear.checkers import (
    RATIONAL_LINEAR_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.rational_linear.operations import (
    compute_rational_inconsistency,
    compute_rational_solution,
)
from jacobian.operation_bindings import inline_operation
from jacobian.operation_declarations import OperationDeclaration, OperationDeclarations


def rational_linear_operations() -> OperationDeclarations:
    operations = (
        inline_operation(
            OperationDeclaration(
                operation_id="linear.rational_solution.compute",
                version="2",
                title="Compute an exact rational solution",
                description="Return one total bounded rational solution candidate inline.",
                request_type=LinearRationalSolutionFindRequest,
                result_type=LinearRationalSolutionResult,
                execute=compute_rational_solution,
                tags=("linear-algebra", "rational", "solution", "exact"),
                examples=(
                    example(
                        "identity_solution",
                        "Solve a one-variable identity system.",
                        {
                            "system": {
                                "variables": ["x"],
                                "coefficients": {
                                    "entries": [[{"num": "1", "den": "1"}]]
                                },
                                "rhs": [{"num": "2", "den": "1"}],
                            }
                        },
                    ),
                ),
            ),
        ),
        inline_operation(
            OperationDeclaration(
                operation_id="linear.rational_inconsistency.compute",
                version="2",
                title="Compute an exact rational inconsistency witness",
                description="Return one normalized left witness inline when the system is inconsistent.",
                request_type=LinearRationalInconsistencyFindRequest,
                result_type=LinearRationalInconsistencyResult,
                execute=compute_rational_inconsistency,
                tags=("linear-algebra", "rational", "inconsistency", "exact"),
            ),
        ),
    )
    return operations


__all__ = ["rational_linear_operations"]

CHECKER_DECLARATIONS = RATIONAL_LINEAR_EXACT_REPLAY_CHECKERS
