"""Installation bundle for exact finite-integer-set operations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.finite_sets.set_cardinality import SET_CARDINALITY_OPERATIONS
from jacobian.domains.finite_sets.set_operations import SET_OPERATION_OPERATIONS
from jacobian.domains.finite_sets.set_predicates import SET_PREDICATE_OPERATIONS
from jacobian.operations import (
    DomainDiagnostics,
    DomainSemantics,
)


def build_finite_set_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="finite_sets",
        schema_namespace="jacobian.finite-sets",
        semantics=DomainSemantics(
            name="jacobian.exact-finite-integer-sets",
            version="1",
            definition={
                "description": "Finite sets of canonical integers with exact operations",
                "element_type": "canonical integer",
                "max_set_size": 128,
            },
        ),
        operations=(
            *SET_OPERATION_OPERATIONS,
            *SET_PREDICATE_OPERATIONS,
            *SET_CARDINALITY_OPERATIONS,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_FINITE_SET_REQUEST",
                stage="finite_set_input_validation",
                message="Input does not satisfy the finite-integer-set contract.",
                hint="Use canonical integer strings and inspect the operation's set schema.",
            )
        ),
    )
