"""Installation bundle for bounded graph optimization."""

from __future__ import annotations

from jacobian.domains.graph_optimization.checkers import (
    GRAPH_SEARCH_EXACT_REPLAY_CHECKERS,
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
from jacobian.operation_declarations import OperationDeclarations


def build_graph_optimization_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (
        CHROMATIC_NUMBER_OPERATION,
        *FINITE_GRAPH_OPTIMIZATION_OPERATIONS,
        HAMILTONIAN_PATH_OPERATION,
        MINIMUM_SPANNING_TREE_OPERATION,
        CLIQUE_NUMBER_OPERATION,
        INDEPENDENCE_NUMBER_OPERATION,
    )


CHECKER_DECLARATIONS = GRAPH_SEARCH_EXACT_REPLAY_CHECKERS
