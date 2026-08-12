"""Installation bundle for exact finite integer-sequence capabilities."""

from __future__ import annotations

import platform

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.sequences.aggregates import SEQUENCE_AGGREGATE_CAPABILITIES
from jacobian.domains.sequences.predicates import SEQUENCE_PREDICATE_CAPABILITIES
from jacobian.domains.sequences.search import SEQUENCE_SEARCH_CAPABILITIES
from jacobian.domains.sequences.statistics import SEQUENCE_STATISTIC_CAPABILITIES
from jacobian.domains.sequences.transforms import SEQUENCE_TRANSFORM_CAPABILITIES
from jacobian.operations import (
    DomainDiagnostics,
    DomainSemantics,
)
from jacobian.provider_runtime import jacobian_provider_runtime


def build_sequence_bundle() -> DomainBundle:
    """Build this domain-owned installation unit explicitly."""
    return DomainBundle(
        domain_id="sequences",
        schema_namespace="jacobian.sequences",
        semantics=DomainSemantics(
            name="jacobian.exact-finite-integer-sequences",
            version="1",
            definition={
                "description": "Finite sequences of canonical integers with exact operations",
                "element_type": "canonical integer",
                "max_sequence_length": 256,
            },
        ),
        provider_runtime=jacobian_provider_runtime(
            "jacobian.sequences",
            features=("exact-integer-sequences",),
        ),
        backend_version=f"python-{platform.python_version()}",
        capabilities=(
            *SEQUENCE_AGGREGATE_CAPABILITIES,
            *SEQUENCE_STATISTIC_CAPABILITIES,
            *SEQUENCE_TRANSFORM_CAPABILITIES,
            *SEQUENCE_PREDICATE_CAPABILITIES,
            *SEQUENCE_SEARCH_CAPABILITIES,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_SEQUENCE_REQUEST",
                stage="sequence_input_validation",
                message="Input does not satisfy the finite-integer-sequence contract.",
                hint="Use canonical integer strings and inspect the operation's sequence schema.",
            )
        ),
    )
