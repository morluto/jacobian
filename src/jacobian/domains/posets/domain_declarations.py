"""Bounded exact finite-poset operation declarations."""

from jacobian.domains.posets.checkers import FINITE_POSET_EXACT_REPLAY_CHECKERS
from jacobian.domains.posets.operations import FINITE_POSET_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def finite_poset_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return FINITE_POSET_OPERATIONS


__all__ = ["finite_poset_operations"]

CHECKER_DECLARATIONS = FINITE_POSET_EXACT_REPLAY_CHECKERS
