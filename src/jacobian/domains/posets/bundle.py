"""Explicit bundle for bounded exact finite posets."""

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.posets.checkers import FINITE_POSET_EXACT_REPLAY_CHECKERS
from jacobian.domains.posets.operations import FINITE_POSET_CAPABILITIES
from jacobian.operations import DomainBundle, DomainDiagnostics, DomainSemantics
from jacobian.provider_runtime import NETWORKX_VERSION, known_provider_runtime


def build_finite_poset_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="poset",
        schema_namespace="jacobian.poset",
        semantics=DomainSemantics(
            name="jacobian.finite-poset",
            version="1",
            definition={
                "description": (
                    "bounded labelled finite partial orders, exact extremal "
                    "certificates, incidence values, and complete ideal recurrences"
                ),
                "carrier": "canonical ASCII labels in lexicographic order",
                "strict_order": "complete irreflexive transitive closure",
                "cover_relation": "unique transitive reduction of the strict order",
                "width": "Dilworth antichain and chain-cover witnesses",
                "linear_extensions": "complete order-ideal recurrence table",
                "mobius": "explicit complete-matrix or selected-interval scope",
                "excluded": (
                    "infinite posets, approximate extension counts, unlabeled "
                    "isomorphism, lattices, and order-dimension claims"
                ),
            },
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.networkx",
            features=(
                "finite-poset",
                "dag-transitive-closure",
                "dag-transitive-reduction",
                "bipartite-maximum-matching",
                "exact-subset-dp",
            ),
        ),
        backend_version=f"networkx-{NETWORKX_VERSION}",
        capabilities=FINITE_POSET_CAPABILITIES,
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_FINITE_POSET_REQUEST",
                stage="finite_poset_input_validation",
                message="Input does not satisfy the bounded finite-poset contract.",
                hint=(
                    "Declare unique labels and either exact cover edges or the "
                    "complete comparable-pair relation under an explicit reflexive policy."
                ),
            )
        ),
        checker_declarations=FINITE_POSET_EXACT_REPLAY_CHECKERS,
    )


__all__ = ["build_finite_poset_bundle"]
