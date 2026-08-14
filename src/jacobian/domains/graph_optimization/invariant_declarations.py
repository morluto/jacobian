"""Finite simple-graph invariant operation declarations."""

from __future__ import annotations

from jacobian.domains.graph_optimization.checkers import (
    GRAPH_INVARIANT_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.graph_optimization.distance_matrix import (
    DISTANCE_MATRIX_OPERATION,
)
from jacobian.domains.graph_optimization.graph6 import (
    GRAPH6_CHECKER_DECLARATIONS,
    GRAPH6_OPERATIONS,
)
from jacobian.domains.graph_optimization.invariants import (
    EXACT_GRAPH_INVARIANT_OPERATIONS,
)
from jacobian.operation_declarations import OperationDeclarations


def graph_invariant_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (
        *GRAPH6_OPERATIONS,
        DISTANCE_MATRIX_OPERATION,
        *EXACT_GRAPH_INVARIANT_OPERATIONS,
    )


CHECKER_DECLARATIONS = (
    *GRAPH6_CHECKER_DECLARATIONS,
    *GRAPH_INVARIANT_EXACT_REPLAY_CHECKERS,
)
