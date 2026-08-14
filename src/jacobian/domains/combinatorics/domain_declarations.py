"""Exact combinatorics operation declarations."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.combinatorics.counting import COUNTING_OPERATIONS
from jacobian.domains.combinatorics.difference_sets import DIFFERENCE_SET_OPERATIONS
from jacobian.domains.combinatorics.partitions import PARTITION_OPERATIONS
from jacobian.domains.combinatorics.recurrence import RECURRENCE_OPERATIONS
from jacobian.operation_declarations import OperationDeclarations, with_invalid_request


def combinatorics_operations() -> OperationDeclarations:
    """Build this domain-owned installation unit explicitly."""
    return with_invalid_request(
        (
            *COUNTING_OPERATIONS,
            *PARTITION_OPERATIONS,
            *RECURRENCE_OPERATIONS,
            *DIFFERENCE_SET_OPERATIONS,
        ),
        OperationDiagnostic(
            code="INVALID_COMBINATORICS_REQUEST",
            stage="combinatorics_input_validation",
            message="Input does not satisfy the exact combinatorics contract.",
            hint="Provide bounded non-negative integers within each operation's limits.",
        ),
    )
