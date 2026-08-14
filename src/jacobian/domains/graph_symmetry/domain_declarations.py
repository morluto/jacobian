"""Exact graph-symmetry operation declarations."""

from jacobian.domains.graph_symmetry.checkers import (
    GRAPH_SYMMETRY_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.graph_symmetry.operations import GRAPH_SYMMETRY_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def graph_symmetry_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return GRAPH_SYMMETRY_OPERATIONS


__all__ = ["graph_symmetry_operations"]

CHECKER_DECLARATIONS = GRAPH_SYMMETRY_EXACT_REPLAY_CHECKERS
