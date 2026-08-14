"""Installation bundle for bounded graph optimization."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
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
from jacobian.operations import (
    DomainDiagnostics,
    DomainSemantics,
)


def build_graph_optimization_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="graph_optimization",
        schema_namespace="jacobian.graph-optimization",
        semantics=DomainSemantics(
            name="jacobian.bounded-graph-optimization",
            version="1",
            definition={
                "description": (
                    "Bounded exact graph-optimization search over finite simple "
                    "undirected graphs with explicit wall-clock budgets"
                ),
                "graph_class": "finite simple undirected",
                "default_max_order": 32,
                "independence_number_max_order": 128,
                "default_max_edges": 496,
                "budget": "explicit wall_seconds per request",
                "search_budget": (
                    "finite optimizers bind their own operation-specific size limits"
                ),
                "conventions": {
                    "minimum_spanning_tree": (
                        "minimum total exact rational edge weight; empty and "
                        "disconnected graphs have no spanning tree"
                    ),
                    "domination": "ordinary closed-neighborhood domination",
                    "saturation_number": "minimum cardinality maximal matching",
                    "induced_forest": "empty induced graph allowed",
                    "induced_tree": (
                        "nonempty connected acyclic; empty source has optimum zero"
                    ),
                    "induced_bipartite": "empty induced graph allowed",
                    "hamiltonian_path": (
                        "spanning simple path; empty graph has the empty path"
                    ),
                    "clique_number": "maximum complete vertex subset",
                    "independence_number": "maximum edge-free vertex subset",
                },
                "timeout_or_cancellation": (
                    "UNKNOWN result with a feasible incumbent and explicit bounds"
                ),
                "minimum_spanning_tree_certificate": (
                    "every non-tree edge is no lighter than the maximum-weight "
                    "edge on its fundamental tree path"
                ),
                "minimum_spanning_tree_ties": (
                    "canonical weight-and-endpoint edge insertion before maintained "
                    "Kruskal selection"
                ),
            },
        ),
        operations=(
            CHROMATIC_NUMBER_OPERATION,
            *FINITE_GRAPH_OPTIMIZATION_OPERATIONS,
            HAMILTONIAN_PATH_OPERATION,
            MINIMUM_SPANNING_TREE_OPERATION,
            CLIQUE_NUMBER_OPERATION,
            INDEPENDENCE_NUMBER_OPERATION,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_CHROMATIC_NUMBER_REQUEST",
                stage="request_validation",
                message=(
                    "The complete chromatic-number request is invalid: validation failed"
                ),
                hint=(
                    "Supply a simple graph with unique vertices and undirected "
                    "edges, plus an optional wall_seconds budget from 1 to 120."
                ),
            )
        ),
        checker_declarations=GRAPH_SEARCH_EXACT_REPLAY_CHECKERS,
    )
