"""Bounded graph-optimization operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.graph_optimization._chromatic_number import (
    CHROMATIC_NUMBER_OPERATION,
)
from jacobian.math.graph_optimization._distance_matrix import (
    DISTANCE_MATRIX_OPERATION,
)
from jacobian.math.graph_optimization._finite_optimization import (
    FINITE_GRAPH_OPTIMIZATION_OPERATIONS,
)
from jacobian.math.graph_optimization._hamiltonian_path import (
    HAMILTONIAN_PATH_OPERATION,
)
from jacobian.math.graph_optimization._independence import (
    INDEPENDENCE_NUMBER_OPERATION,
)
from jacobian.math.graph_optimization._invariants import (
    CLIQUE_NUMBER_OPERATION,
    EXACT_GRAPH_INVARIANT_OPERATIONS,
)
from jacobian.math.graph_optimization._minimum_spanning_tree import (
    MINIMUM_SPANNING_TREE_OPERATION,
)

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    CHROMATIC_NUMBER_OPERATION,
    *FINITE_GRAPH_OPTIMIZATION_OPERATIONS,
    HAMILTONIAN_PATH_OPERATION,
    MINIMUM_SPANNING_TREE_OPERATION,
    CLIQUE_NUMBER_OPERATION,
    INDEPENDENCE_NUMBER_OPERATION,
    DISTANCE_MATRIX_OPERATION,
    *EXACT_GRAPH_INVARIANT_OPERATIONS,
)
