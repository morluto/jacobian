"""Explicit bundle for exact declared graph-symmetry actions."""

from jacobian.domains.graph_symmetry.checkers import (
    GRAPH_SYMMETRY_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.graph_symmetry.operations import GRAPH_SYMMETRY_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def build_graph_symmetry_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return GRAPH_SYMMETRY_OPERATIONS


__all__ = ["build_graph_symmetry_bundle"]

CHECKER_DECLARATIONS = GRAPH_SYMMETRY_EXACT_REPLAY_CHECKERS
