"""Installation bundle for exact finite integer-sequence operations."""

from __future__ import annotations

from jacobian.domains.sequences.aggregates import SEQUENCE_AGGREGATE_OPERATIONS
from jacobian.domains.sequences.predicates import SEQUENCE_PREDICATE_OPERATIONS
from jacobian.domains.sequences.search import SEQUENCE_SEARCH_OPERATIONS
from jacobian.domains.sequences.statistics import SEQUENCE_STATISTIC_OPERATIONS
from jacobian.domains.sequences.transforms import SEQUENCE_TRANSFORM_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def build_sequence_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (
        *SEQUENCE_AGGREGATE_OPERATIONS,
        *SEQUENCE_STATISTIC_OPERATIONS,
        *SEQUENCE_TRANSFORM_OPERATIONS,
        *SEQUENCE_PREDICATE_OPERATIONS,
        *SEQUENCE_SEARCH_OPERATIONS,
    )


CHECKER_DECLARATIONS = ()
