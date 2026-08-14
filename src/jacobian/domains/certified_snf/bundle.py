"""Explicit bundle for transformation-certified Smith normal forms."""

from jacobian.domains.certified_snf.checkers import CERTIFIED_SNF_EXACT_REPLAY_CHECKERS
from jacobian.domains.certified_snf.operations import CERTIFIED_SNF_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def build_certified_snf_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return CERTIFIED_SNF_OPERATIONS


__all__ = ["build_certified_snf_bundle"]

CHECKER_DECLARATIONS = CERTIFIED_SNF_EXACT_REPLAY_CHECKERS
