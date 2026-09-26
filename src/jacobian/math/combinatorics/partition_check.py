"""Native checking and canonical Ferrers data for integer partitions."""

from jacobian.catalog.models import OperationDomainValidationError, OperationResourceAdmissionError
from jacobian.math.combinatorics._partition_models import (
    MAX_PARTITION_ITEM,
    MAX_PARTITION_SIZE,
    IncreasingPartsObstruction,
    NonpositivePartObstruction,
    PartitionCheckResult,
    PartitionFound,
    PartitionRejected,
)
from jacobian.math.combinatorics.symmetric_functions.values import IntegerPartition


def check_integer_partition(parts: tuple[int, ...]) -> PartitionCheckResult:
    """Classify a bounded integer tuple and return its canonical Ferrers data."""
    if type(parts) is not tuple or len(parts) > MAX_PARTITION_SIZE:
        raise OperationDomainValidationError(
            location=("parts",),
            code="combinatorics.partition_candidate_shape",
            message="candidate must be a bounded tuple of exact integers",
        )
    if any(type(part) is not int or abs(part) > MAX_PARTITION_ITEM for part in parts):
        raise OperationDomainValidationError(
            location=("parts",),
            code="combinatorics.partition_candidate_integer",
            message="candidate parts must be exact JSON-safe integers",
        )
    previous: int | None = None
    for index, part in enumerate(parts):
        if part <= 0:
            return PartitionCheckResult(outcome=PartitionRejected(
                parts=parts,
                obstruction=NonpositivePartObstruction(index=index, value=part),
            ))
        if previous is not None and previous < part:
            return PartitionCheckResult(outcome=PartitionRejected(
                parts=parts,
                obstruction=IncreasingPartsObstruction(
                    index=index, previous_value=previous, value=part
                ),
            ))
        previous = part
    if sum(parts) > MAX_PARTITION_SIZE:
        raise OperationResourceAdmissionError(
            location=("parts",),
            code="combinatorics.partition_candidate_size",
            message="valid partition candidate exceeds the supported size",
        )
    partition = IntegerPartition(parts=parts)
    conjugate = tuple(
        sum(part >= column for part in parts)
        for column in range(1, (parts[0] if parts else 0) + 1)
    )
    cells = tuple(
        (row, column)
        for row, part in enumerate(parts, start=1)
        for column in range(1, part + 1)
    )
    return PartitionCheckResult(outcome=PartitionFound._from_checked(
        partition=partition, conjugate=IntegerPartition(parts=conjugate), cells=cells
    ))


__all__ = ["check_integer_partition"]
