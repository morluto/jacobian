"""Validated real-analysis operation declarations."""

from jacobian.domains.analysis.operations import POINT_ENCLOSURE_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def real_analysis_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return POINT_ENCLOSURE_OPERATIONS


__all__ = ["real_analysis_operations"]

CHECKER_DECLARATIONS = ()
