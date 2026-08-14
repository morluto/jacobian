"""Installation bundle for finite simple-graph invariants."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
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
from jacobian.operations import DomainDiagnostics, DomainSemantics


def build_graph_invariant_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="graph_invariants",
        schema_namespace="jacobian.graph-invariants",
        semantics=DomainSemantics(
            name="jacobian.finite-simple-graph-invariants",
            version="1",
            definition={
                "graph_class": "finite simple undirected",
                "maximum_order": 32,
                "maximum_edges": 496,
                "distance_matrix_maximum_order": 64,
                "distance_matrix_maximum_edges": 2_016,
                "exact_computations": [
                    "distance_matrix",
                    "girth",
                    "diameter",
                    "edge_connectivity",
                    "vertex_connectivity",
                    "is_eulerian",
                    "spanning_tree_count",
                    "maximum_matching",
                ],
                "spanning_tree_arithmetic": "exact SymPy integer determinant",
            },
        ),
        operations=(
            *GRAPH6_OPERATIONS,
            DISTANCE_MATRIX_OPERATION,
            *EXACT_GRAPH_INVARIANT_OPERATIONS,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_GRAPH_INVARIANT_REQUEST",
                stage="graph_invariant_input_validation",
                message="Input does not satisfy the bounded graph invariant contract.",
                hint="Supply a canonical simple graph with at most 32 vertices.",
            )
        ),
        checker_declarations=(
            *GRAPH6_CHECKER_DECLARATIONS,
            *GRAPH_INVARIANT_EXACT_REPLAY_CHECKERS,
        ),
    )
