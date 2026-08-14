"""Bounded graph-optimization operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.graph_optimization.checkers import (
    GRAPH_SEARCH_AUTHORIZED_CHECKERS,
)
from jacobian.domains.graph_optimization.chromatic_number import (
    CHROMATIC_NUMBER_OPERATION,
)
from jacobian.domains.graph_optimization.finite_optimization import (
    FINITE_GRAPH_OPTIMIZATION_OPERATIONS,
)
from jacobian.domains.graph_optimization.hamiltonian_path import (
    HAMILTONIAN_PATH_OPERATION,
)
from jacobian.domains.graph_optimization.independence import (
    INDEPENDENCE_NUMBER_OPERATION,
)
from jacobian.domains.graph_optimization.invariants import (
    CLIQUE_NUMBER_OPERATION,
)
from jacobian.domains.graph_optimization.minimum_spanning_tree import (
    MINIMUM_SPANNING_TREE_OPERATION,
)
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def graph_optimization_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            CHROMATIC_NUMBER_OPERATION,
            *FINITE_GRAPH_OPTIMIZATION_OPERATIONS,
            HAMILTONIAN_PATH_OPERATION,
            MINIMUM_SPANNING_TREE_OPERATION,
            CLIQUE_NUMBER_OPERATION,
            INDEPENDENCE_NUMBER_OPERATION,
        ),
        OperationDiagnostic(
            code="INVALID_CHROMATIC_NUMBER_REQUEST",
            stage="request_validation",
            message="The complete chromatic-number request is invalid.",
            hint="Supply a canonical bounded simple graph.",
        ),
    )


AUTHORIZED_CHECKERS = GRAPH_SEARCH_AUTHORIZED_CHECKERS
