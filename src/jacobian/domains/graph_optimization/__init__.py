"""Bounded graph-optimization operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_invariant_operations", "graph_optimization_operations"]


def graph_optimization_operations() -> MathTools:
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
    from jacobian.domains.graph_optimization.invariants import CLIQUE_NUMBER_OPERATION
    from jacobian.domains.graph_optimization.minimum_spanning_tree import (
        MINIMUM_SPANNING_TREE_OPERATION,
    )

    return (
        CHROMATIC_NUMBER_OPERATION,
        *FINITE_GRAPH_OPTIMIZATION_OPERATIONS,
        HAMILTONIAN_PATH_OPERATION,
        MINIMUM_SPANNING_TREE_OPERATION,
        CLIQUE_NUMBER_OPERATION,
        INDEPENDENCE_NUMBER_OPERATION,
    )


def graph_invariant_operations() -> MathTools:
    from jacobian.domains.graph_optimization.distance_matrix import (
        DISTANCE_MATRIX_OPERATION,
    )
    from jacobian.domains.graph_optimization.graph6 import GRAPH6_OPERATIONS
    from jacobian.domains.graph_optimization.invariants import (
        EXACT_GRAPH_INVARIANT_OPERATIONS,
    )

    return (
        *GRAPH6_OPERATIONS,
        DISTANCE_MATRIX_OPERATION,
        *EXACT_GRAPH_INVARIANT_OPERATIONS,
    )
