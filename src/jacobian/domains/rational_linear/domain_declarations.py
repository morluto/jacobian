"""Domain-owned rational-linear operation declarations."""

from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalInconsistencyResult,
    LinearRationalSolutionFindRequest,
    LinearRationalSolutionResult,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains._examples import example
from jacobian.domains.rational_linear.checkers import (
    RATIONAL_LINEAR_AUTHORIZED_CHECKERS,
)
from jacobian.domains.rational_linear.operations import (
    compute_rational_inconsistency,
    compute_rational_solution,
)
from jacobian.operation_declarations import (
    OperationDeclaration,
    OperationDeclarations,
    inline_operation,
    with_invalid_request,
)


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
    return with_invalid_request(
        operations,
        OperationDiagnostic(
            code="INVALID_RATIONAL_LINEAR_REQUEST",
            stage="rational_linear_input_validation",
            message="Input does not satisfy the exact rational-linear contract.",
            hint="Use canonical rational components and bounded dimensions.",
        ),
    )


__all__ = ["rational_linear_operations"]

AUTHORIZED_CHECKERS = RATIONAL_LINEAR_AUTHORIZED_CHECKERS
