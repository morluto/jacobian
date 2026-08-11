"""Installation bundle for exact finite-integer-set capabilities."""

from __future__ import annotations

import platform

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains.finite_sets.set_cardinality import SET_CARDINALITY_CAPABILITIES
from jacobian.domains.finite_sets.set_operations import SET_OPERATION_CAPABILITIES
from jacobian.domains.finite_sets.set_predicates import SET_PREDICATE_CAPABILITIES
from jacobian.operations import (
    DomainBundle,
    DomainDiagnostics,
    DomainSemantics,
)
from jacobian.provider_runtime import jacobian_provider_runtime


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
                "assurance": "computed; no independent checker",
            },
        ),
        provider_runtime=jacobian_provider_runtime(
            "jacobian.finite-sets",
            features=("exact-finite-sets",),
        ),
        backend_version=f"python-{platform.python_version()}",
        capabilities=(
            *SET_OPERATION_CAPABILITIES,
            *SET_PREDICATE_CAPABILITIES,
            *SET_CARDINALITY_CAPABILITIES,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_FINITE_SET_REQUEST",
                stage="finite_set_input_validation",
                message="Input does not satisfy the finite-integer-set contract.",
                hint="Use canonical integer strings and inspect the operation's set schema.",
            )
        ),
    )
