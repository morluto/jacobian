"""Explicit bundle for bounded exact finite posets."""

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.posets.checkers import FINITE_POSET_EXACT_REPLAY_CHECKERS
from jacobian.domains.posets.operations import FINITE_POSET_OPERATIONS
from jacobian.operations import DomainDiagnostics, DomainSemantics


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
        operations=FINITE_POSET_OPERATIONS,
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
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
