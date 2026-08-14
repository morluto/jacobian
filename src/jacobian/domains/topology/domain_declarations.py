"""Exact finite simplicial-topology operation declarations."""

from __future__ import annotations

from jacobian.domains.topology.checkers import TOPOLOGY_EXACT_REPLAY_CHECKERS
from jacobian.domains.topology.operations import TOPOLOGY_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def topology_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return TOPOLOGY_OPERATIONS


__all__ = ["topology_operations"]

CHECKER_DECLARATIONS = TOPOLOGY_EXACT_REPLAY_CHECKERS
