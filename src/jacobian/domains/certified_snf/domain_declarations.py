"""Operation declarations for transformation-certified Smith normal forms."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.certified_snf.operations import CERTIFIED_SNF_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def certified_snf_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        CERTIFIED_SNF_OPERATIONS,
        OperationDiagnostic(
            code="INVALID_CERTIFIED_SMITH_REQUEST",
            stage="certified_smith_input_validation",
            message="Input does not satisfy the bounded certified-Smith contract.",
            hint=(
                "Supply a nonempty matrix of at most 16 by 16 canonical integer "
                "strings, each containing at most 32 decimal digits."
            ),
        ),
    )


__all__ = ["certified_snf_operations"]
