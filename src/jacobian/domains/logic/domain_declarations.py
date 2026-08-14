"""Explicit declarations for atomic logic operations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.logic.operations import LOGIC_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def logic_operations() -> OperationDeclarations:
    """Build the stateless logic operation family."""

    return with_invalid_request(
        LOGIC_OPERATIONS,
        OperationDiagnostic(
            code="INVALID_LOGIC_REQUEST",
            stage="logic_input_validation",
            message="Input does not satisfy the bounded logic operation contract.",
            hint="Use the exact typed inline values shown by the operation schema.",
        ),
    )


__all__ = ["logic_operations"]
