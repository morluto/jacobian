"""Exact bounded standard Young tableau enumeration."""

from __future__ import annotations

from collections.abc import Iterator

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.algebraic.operations import (
    standard_young_tableaux_count,
)
from jacobian.math.combinatorics.standard_tableaux._models import (
    MAX_CONSTRUCTION_WORK_CELLS,
    MAX_ENUMERATED_CELLS,
    MAX_RESULT_CELLS,
    MAX_STANDARD_TABLEAUX,
    StandardTableauEnumerationResult,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    StandardYoungTableau,
)


def _remaining_shape(parts: tuple[int, ...], row: int) -> tuple[int, ...]:
    reduced = list(parts)
    reduced[row] -= 1
    if reduced[row] == 0:
        reduced.pop(row)
    return tuple(reduced)


def _tableau_rows(
    parts: tuple[int, ...],
) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Yield tableaux by removing and restoring the largest entry at corners."""
    size = sum(parts)
    if size == 0:
        yield ()
        return

    for row in range(len(parts)):
        next_length = parts[row + 1] if row + 1 < len(parts) else 0
        if parts[row] == next_length:
            continue
        smaller = _remaining_shape(parts, row)
        for previous in _tableau_rows(smaller):
            rows = [list(values) for values in previous]
            while len(rows) <= row:
                rows.append([])
            rows[row].append(size)
            yield tuple(tuple(values) for values in rows)


def enumerate_standard_young_tableaux(
    partition: IntegerPartition,
) -> StandardTableauEnumerationResult:
    """Return all standard tableaux after count and output admission."""
    if not isinstance(partition, IntegerPartition):
        raise OperationDomainValidationError(
            location=("partition",),
            code="standard_tableaux.partition_type",
            message="partition must be an IntegerPartition",
        )
    try:
        partition = IntegerPartition.model_validate(partition.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("partition",),
            code="standard_tableaux.invalid_partition",
            message="partition is not a valid canonical integer partition",
        ) from exc
    request_checkpoint("before standard-tableau enumeration")
    size = sum(partition.parts)
    count = standard_young_tableaux_count(partition)
    if count > MAX_STANDARD_TABLEAUX:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="standard_tableaux.count_bound",
            message=(
                "the complete standard-tableau family exceeds the admitted "
                f"count of {MAX_STANDARD_TABLEAUX}"
            ),
        )
    if count * size > MAX_ENUMERATED_CELLS:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="standard_tableaux.cell_bound",
            message=(
                "the complete standard-tableau family exceeds the admitted "
                f"aggregate cell count of {MAX_ENUMERATED_CELLS}"
            ),
        )
    corner_work = count * size * (size + 1)
    sort_work = count * (count - 1).bit_length() * size
    if corner_work + sort_work > MAX_CONSTRUCTION_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="standard_tableaux.work_bound",
            message=(
                "the complete standard-tableau family exceeds the admitted "
                "corner-construction work envelope"
            ),
        )
    result_cells = count * (size + 1) + size
    if result_cells > MAX_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="standard_tableaux.output_bound",
            message=(
                "the complete standard-tableau family exceeds the admitted "
                f"result cell count of {MAX_RESULT_CELLS}"
            ),
        )

    rows_list = []
    for index, value in enumerate(_tableau_rows(partition.parts)):
        if index % 128 == 0:
            request_checkpoint("during standard-tableau enumeration")
        rows_list.append(value)
    request_checkpoint("before standard-tableau sorting")
    rows = tuple(sorted(rows_list))
    tableaux_list = []
    for index, value in enumerate(rows):
        if index % 128 == 0:
            request_checkpoint("during standard-tableau materialization")
        tableaux_list.append(StandardYoungTableau(rows=value))
    request_checkpoint("before standard-tableau result construction")
    tableaux = tuple(tableaux_list)
    return StandardTableauEnumerationResult(
        partition=partition,
        tableaux=tableaux,
    )


__all__ = ["enumerate_standard_young_tableaux"]
