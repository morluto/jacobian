"""Operation declarations for transformation-certified Smith normal forms."""

from jacobian.domains.certified_snf.checkers import CERTIFIED_SNF_EXACT_REPLAY_CHECKERS
from jacobian.domains.certified_snf.operations import CERTIFIED_SNF_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def certified_snf_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return CERTIFIED_SNF_OPERATIONS


__all__ = ["certified_snf_operations"]

CHECKER_DECLARATIONS = CERTIFIED_SNF_EXACT_REPLAY_CHECKERS
