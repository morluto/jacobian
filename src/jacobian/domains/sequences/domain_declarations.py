"""Exact finite integer-sequence operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.sequences.aggregates import SEQUENCE_AGGREGATE_OPERATIONS
from jacobian.domains.sequences.predicates import SEQUENCE_PREDICATE_OPERATIONS
from jacobian.domains.sequences.search import SEQUENCE_SEARCH_OPERATIONS
from jacobian.domains.sequences.statistics import SEQUENCE_STATISTIC_OPERATIONS
from jacobian.domains.sequences.transforms import SEQUENCE_TRANSFORM_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def sequence_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            *SEQUENCE_AGGREGATE_OPERATIONS,
            *SEQUENCE_STATISTIC_OPERATIONS,
            *SEQUENCE_TRANSFORM_OPERATIONS,
            *SEQUENCE_PREDICATE_OPERATIONS,
            *SEQUENCE_SEARCH_OPERATIONS,
        ),
        OperationDiagnostic(
            code="INVALID_SEQUENCE_REQUEST",
            stage="sequence_input_validation",
            message="Input does not satisfy the finite-integer-sequence contract.",
            hint=(
                "Use canonical integer strings and inspect the operation's sequence schema."
            ),
        ),
    )
