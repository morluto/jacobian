"""Validated real-analysis domain bundle."""

from jacobian.domains.analysis.operations import POINT_ENCLOSURE_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def build_real_analysis_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return POINT_ENCLOSURE_OPERATIONS


__all__ = ["build_real_analysis_bundle"]

CHECKER_DECLARATIONS = ()
