"""Exact bounded standard Young tableau enumeration."""

from __future__ import annotations

from collections.abc import Iterator

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.algebraic.operations import (
    standard_young_tableaux_count,
)
from jacobian.math.combinatorics.standard_tableaux._models import (
    MAX_CONSTRUCTION_WORK_CELLS,
    MAX_ENUMERATED_CELLS,
    MAX_RESULT_BYTES,
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
    estimated_bytes = count * (10 * size + 64) + 128
    if estimated_bytes > MAX_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="standard_tableaux.output_bound",
            message=(
                "the complete standard-tableau family exceeds the admitted "
                f"result size of {MAX_RESULT_BYTES} bytes"
            ),
        )

    rows = tuple(sorted(_tableau_rows(partition.parts)))
    tableaux = tuple(StandardYoungTableau(rows=value) for value in rows)
    return StandardTableauEnumerationResult(
        partition=partition,
        tableaux=tableaux,
    )


__all__ = ["enumerate_standard_young_tableaux"]
