"""Installation bundle for bounded graph optimization."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.graph_optimization.checkers import (
    GRAPH_SEARCH_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.graph_optimization.chromatic_number import (
    CHROMATIC_NUMBER_CAPABILITY,
)
from jacobian.domains.graph_optimization.finite_optimization import (
    FINITE_GRAPH_OPTIMIZATION_CAPABILITIES,
)
from jacobian.domains.graph_optimization.hamiltonian_path import (
    HAMILTONIAN_PATH_CAPABILITY,
)
from jacobian.domains.graph_optimization.invariants import (
    BOUNDED_GRAPH_INVARIANT_CAPABILITIES,
)
from jacobian.domains.graph_optimization.minimum_spanning_tree import (
    MINIMUM_SPANNING_TREE_CAPABILITY,
)
from jacobian.operations import (
    DomainBundle,
    DomainDiagnostics,
    DomainSemantics,
)
from jacobian.provider_runtime import (
    NETWORKX_VERSION,
    SYMPY_VERSION,
    Z3_SOLVER_VERSION,
    composite_provider_runtime,
    known_provider_runtime,
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
                "max_order": 32,
                "max_edges": 496,
                "budget": "explicit wall_seconds per request",
                "search_budget": (
                    "finite optimizers also bind max_order and max_solver_calls"
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
                    "UNKNOWN partial result with preserved bounds and tested obligations"
                ),
                "minimum_spanning_tree_certificate": (
                    "every non-tree edge is no lighter than the maximum-weight "
                    "edge on its fundamental tree path"
                ),
                "minimum_spanning_tree_ties": (
                    "canonical weight-and-endpoint edge insertion before maintained "
                    "Kruskal selection"
                ),
                "assurance": "computed; incomplete search is never a conclusion",
            },
        ),
        provider_runtime=composite_provider_runtime(
            "jacobian.graph-optimization",
            components=(
                known_provider_runtime(
                    "jacobian.z3",
                    features=("bounded-finite-search",),
                ),
                known_provider_runtime(
                    "jacobian.networkx",
                    features=(
                        "graph-witness-validation",
                        "graph-approximations",
                        "exact-rational-minimum-spanning-tree",
                    ),
                ),
                known_provider_runtime(
                    "jacobian.sympy",
                    features=("exact-spanning-tree-determinant",),
                ),
            ),
            features=(
                "bounded-k-colorability",
                "finite-graph-optimization",
                "exact-rational-minimum-spanning-tree",
                "timeout-aware",
            ),
        ),
        backend_version=(
            f"z3-solver-{Z3_SOLVER_VERSION};networkx-{NETWORKX_VERSION};"
            f"sympy-{SYMPY_VERSION}"
        ),
        capabilities=(
            CHROMATIC_NUMBER_CAPABILITY,
            *FINITE_GRAPH_OPTIMIZATION_CAPABILITIES,
            HAMILTONIAN_PATH_CAPABILITY,
            MINIMUM_SPANNING_TREE_CAPABILITY,
            *BOUNDED_GRAPH_INVARIANT_CAPABILITIES,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
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
        scope_description="one bounded simple undirected graph",
        completeness_basis=(
            "Z3 settled every stronger threshold needed to bind the reported optimum"
        ),
        assurance_basis=(
            "bounded Z3 computation with NetworkX witness predicates; an "
            "independent checker is still required for VERIFIED assurance"
        ),
        checker_declarations=GRAPH_SEARCH_EXACT_REPLAY_CHECKERS,
    )
