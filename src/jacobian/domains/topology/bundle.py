"""Explicit bundle for exact finite simplicial topology."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.topology.checkers import TOPOLOGY_EXACT_REPLAY_CHECKERS
from jacobian.domains.topology.operations import TOPOLOGY_OPERATIONS
from jacobian.operations import DomainDiagnostics, DomainSemantics


def build_topology_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="topology",
        schema_namespace="jacobian.topology",
        semantics=DomainSemantics(
            name="jacobian.finite-simplicial-topology",
            version="1",
            definition={
                "description": (
                    "bounded finite abstract simplicial complexes, oriented chain "
                    "complexes, homology over bounded prime fields, and bounded "
                    "transformation-certified integral homology"
                ),
                "vertices": "canonical ASCII labels in lexicographic order",
                "orientation": "the increasing vertex order orients every simplex",
                "faces": "the empty simplex is not stored",
                "isolated_vertices": "represented by singleton maximal facets",
                "homology": "reduced or unreduced convention is explicit",
                "integral_homology": (
                    "free rank, torsion invariant factors, simplex-basis generators, "
                    "bounding chains, and full Smith transformation certificates"
                ),
                "persistent_homology": "not provided",
            },
        ),
        operations=TOPOLOGY_OPERATIONS,
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_FINITE_SIMPLICIAL_TOPOLOGY_REQUEST",
                stage="finite_simplicial_topology_input_validation",
                message="Input does not satisfy the bounded simplicial-topology contract.",
                hint=(
                    "Use unique declared vertices, distinct maximal facets, and a "
                    "bounded prime when requesting F_p homology; integral homology "
                    "has tighter 16-simplex-per-chain-group and total-chain-rank-32 "
                    "bounds."
                ),
            )
        ),
        checker_declarations=TOPOLOGY_EXACT_REPLAY_CHECKERS,
    )


__all__ = ["build_topology_bundle"]
