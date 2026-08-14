"""Validated real-analysis operation declarations."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.analysis.operations import POINT_ENCLOSURE_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def real_analysis_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        POINT_ENCLOSURE_OPERATIONS,
        OperationDiagnostic(
            code="INVALID_REAL_ANALYSIS_REQUEST",
            stage="real_analysis_input_validation",
            message="Input does not satisfy the bounded real-analysis contract.",
            hint="Use a supported function, bounded rational, and declared precision.",
        ),
    )


__all__ = ["real_analysis_operations"]
