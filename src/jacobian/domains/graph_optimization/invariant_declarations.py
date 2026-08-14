"""Finite simple-graph invariant operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.graph_optimization.checkers import (
    GRAPH_INVARIANT_AUTHORIZED_CHECKERS,
)
from jacobian.domains.graph_optimization.distance_matrix import (
    DISTANCE_MATRIX_OPERATION,
)
from jacobian.domains.graph_optimization.graph6 import (
    GRAPH6_AUTHORIZED_CHECKERS,
    GRAPH6_OPERATIONS,
)
from jacobian.domains.graph_optimization.invariants import (
    EXACT_GRAPH_INVARIANT_OPERATIONS,
)
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def graph_invariant_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            *GRAPH6_OPERATIONS,
            DISTANCE_MATRIX_OPERATION,
            *EXACT_GRAPH_INVARIANT_OPERATIONS,
        ),
        OperationDiagnostic(
            code="INVALID_GRAPH_INVARIANT_REQUEST",
            stage="graph_invariant_input_validation",
            message="Input does not satisfy the bounded graph invariant contract.",
            hint="Supply a canonical simple graph with at most 32 vertices.",
        ),
    )


AUTHORIZED_CHECKERS = (
    *GRAPH6_AUTHORIZED_CHECKERS,
    *GRAPH_INVARIANT_AUTHORIZED_CHECKERS,
)
