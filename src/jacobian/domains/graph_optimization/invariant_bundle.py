"""Installation bundle for finite simple-graph invariants."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.graph_optimization.checkers import (
    GRAPH_INVARIANT_EXACT_REPLAY_CHECKERS,
)
from jacobian.domains.graph_optimization.invariants import (
    EXACT_GRAPH_INVARIANT_CAPABILITIES,
)
from jacobian.operations import DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import (
    NETWORKX_VERSION,
    SYMPY_VERSION,
    composite_provider_runtime,
    known_provider_runtime,
)


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
                "exact_computations": [
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
        provider_runtime=composite_provider_runtime(
            "jacobian.graph-invariants",
            components=(
                known_provider_runtime(
                    "jacobian.networkx",
                    features=(
                        "finite-simple-graph",
                        "exact-invariants",
                        "matching-witnesses",
                    ),
                ),
                known_provider_runtime(
                    "jacobian.sympy",
                    features=("exact-spanning-tree-determinant",),
                ),
            ),
            features=(
                "finite-simple-graph",
                "exact-invariants",
                "matching-witnesses",
            ),
        ),
        backend_version=f"networkx-{NETWORKX_VERSION};sympy-{SYMPY_VERSION}",
        capabilities=EXACT_GRAPH_INVARIANT_CAPABILITIES,
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_GRAPH_INVARIANT_REQUEST",
                stage="graph_invariant_input_validation",
                message="Input does not satisfy the bounded graph invariant contract.",
                hint="Supply a canonical simple graph with at most 32 vertices.",
            )
        ),
        checker_declarations=GRAPH_INVARIANT_EXACT_REPLAY_CHECKERS,
    )
