"""Explicit bundle for bounded exact finite posets."""

from jacobian.domains.posets.checkers import FINITE_POSET_EXACT_REPLAY_CHECKERS
from jacobian.domains.posets.operations import FINITE_POSET_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def build_finite_poset_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return FINITE_POSET_OPERATIONS


__all__ = ["build_finite_poset_bundle"]

CHECKER_DECLARATIONS = FINITE_POSET_EXACT_REPLAY_CHECKERS
