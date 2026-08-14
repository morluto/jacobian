"""Exact finite simplicial-topology operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.topology.checkers import TOPOLOGY_AUTHORIZED_CHECKERS
from jacobian.domains.topology.operations import TOPOLOGY_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def topology_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        TOPOLOGY_OPERATIONS,
        OperationDiagnostic(
            code="INVALID_FINITE_SIMPLICIAL_TOPOLOGY_REQUEST",
            stage="finite_simplicial_topology_input_validation",
            message="Input does not satisfy the bounded simplicial-topology contract.",
            hint=(
                "Use unique declared vertices, distinct maximal facets, and a bounded "
                "prime for F_p homology; integral homology has tighter chain bounds."
            ),
        ),
    )


__all__ = ["topology_operations"]

AUTHORIZED_CHECKERS = TOPOLOGY_AUTHORIZED_CHECKERS
