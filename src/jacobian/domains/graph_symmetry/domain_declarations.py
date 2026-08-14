"""Exact graph-symmetry operation declarations."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.graph_symmetry.checkers import (
    GRAPH_SYMMETRY_AUTHORIZED_CHECKERS,
)
from jacobian.domains.graph_symmetry.operations import GRAPH_SYMMETRY_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def graph_symmetry_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        GRAPH_SYMMETRY_OPERATIONS,
        OperationDiagnostic(
            code="INVALID_GRAPH_SYMMETRY_REQUEST",
            stage="graph_symmetry_input_validation",
            message="Input does not satisfy the bounded declared-symmetry contract.",
            hint=(
                "Supply total vertex permutations that preserve every graph edge "
                "and every declared vertex or edge color."
            ),
        ),
    )


__all__ = ["graph_symmetry_operations"]

AUTHORIZED_CHECKERS = GRAPH_SYMMETRY_AUTHORIZED_CHECKERS
