"""Installation bundle for exact finite integer-sequence operations."""

from __future__ import annotations

import platform

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domain_bundles import DomainBundle
from jacobian.domains.sequences.aggregates import SEQUENCE_AGGREGATE_OPERATIONS
from jacobian.domains.sequences.predicates import SEQUENCE_PREDICATE_OPERATIONS
from jacobian.domains.sequences.search import SEQUENCE_SEARCH_OPERATIONS
from jacobian.domains.sequences.statistics import SEQUENCE_STATISTIC_OPERATIONS
from jacobian.domains.sequences.transforms import SEQUENCE_TRANSFORM_OPERATIONS
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
        operations=(
            *SEQUENCE_AGGREGATE_OPERATIONS,
            *SEQUENCE_STATISTIC_OPERATIONS,
            *SEQUENCE_TRANSFORM_OPERATIONS,
            *SEQUENCE_PREDICATE_OPERATIONS,
            *SEQUENCE_SEARCH_OPERATIONS,
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=OperationDiagnostic(
                code="INVALID_SEQUENCE_REQUEST",
                stage="sequence_input_validation",
                message="Input does not satisfy the finite-integer-sequence contract.",
                hint="Use canonical integer strings and inspect the operation's sequence schema.",
            )
        ),
    )
