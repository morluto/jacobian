"""Rational optimization operation declarations."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.optimization.operations import RATIONAL_LINEAR_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def rational_optimization_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        RATIONAL_LINEAR_OPERATIONS,
        OperationDiagnostic(
            code="INVALID_RATIONAL_OPTIMIZATION_REQUEST",
            stage="rational_optimization_input_validation",
            message="Input does not satisfy the rational optimization contract.",
            hint="Use the declared bounded standard-form rational LP model.",
        ),
    )


__all__ = ["rational_optimization_operations"]
