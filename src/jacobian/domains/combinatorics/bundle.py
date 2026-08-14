"""Installation bundle for exact combinatorics operations."""

from __future__ import annotations

from jacobian.domains.combinatorics.checkers import COMBINATORICS_EXACT_REPLAY_CHECKERS
from jacobian.domains.combinatorics.counting import COUNTING_OPERATIONS
from jacobian.domains.combinatorics.difference_sets import DIFFERENCE_SET_OPERATIONS
from jacobian.domains.combinatorics.partitions import PARTITION_OPERATIONS
from jacobian.domains.combinatorics.recurrence import RECURRENCE_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations


def build_combinatorics_bundle() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return (
        *COUNTING_OPERATIONS,
        *PARTITION_OPERATIONS,
        *RECURRENCE_OPERATIONS,
        *DIFFERENCE_SET_OPERATIONS,
    )


CHECKER_DECLARATIONS = COMBINATORICS_EXACT_REPLAY_CHECKERS
