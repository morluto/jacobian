"""Rational optimization operation declarations."""

from jacobian.domains.optimization.checkers import (
    RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.optimization.operations import RATIONAL_LINEAR_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def rational_optimization_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return RATIONAL_LINEAR_OPERATIONS


__all__ = ["rational_optimization_operations"]

CHECKER_DECLARATIONS = RATIONAL_OPTIMIZATION_EXACT_REPLAY_CHECKERS
