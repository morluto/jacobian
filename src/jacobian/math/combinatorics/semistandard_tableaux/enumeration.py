"""Exact bounded enumeration of semistandard Young tableaux."""

from __future__ import annotations

from fractions import Fraction

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.semistandard_tableaux._models import (
    MAX_ENUMERATED_CELLS,
    MAX_ENUMERATION_WORK,
    MAX_RESULT_BYTES,
    MAX_SEMISTANDARD_TABLEAUX,
    SemistandardTableauEnumerationResult,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
)


def semistandard_tableaux_count(partition: IntegerPartition, max_entry: int) -> int:
    """Compute the exact hook-content count for entries in ``1..max_entry``."""
    parts = partition.parts
    if not parts:
        return 1
    width = parts[0]
    column_heights = tuple(
        sum(row_width >= column + 1 for row_width in parts) for column in range(width)
    )
    count = Fraction(1)
    for row, row_width in enumerate(parts):
        for column in range(row_width):
            hook = row_width - column + column_heights[column] - row - 1
            content_factor = max_entry + column - row
            count *= Fraction(content_factor, hook)
            if count == 0:
                return 0
    if count.denominator != 1:
        raise AssertionError("hook-content product must be integral")
    return count.numerator


def _horizontal_strip_predecessors(
    parts: tuple[int, ...], maximum_height: int
) -> tuple[tuple[int, ...], ...]:
    """Return feasible mu with lambda/mu a nonempty horizontal strip.

    The interlacing inequalities ``lambda_i >= mu_i >= lambda_(i+1)`` are
    built into the coordinate ranges. Requiring ``height(mu) <= maximum_height``
    ensures every returned shape has at least one tableau in the remaining
    alphabet (fill row i with i). Thus no emitted branch is a dead search node.
    """
    height = len(parts)
    allowed_rows = min(height, maximum_height)
    if allowed_rows < height - 1:
        return ()

    chosen: list[int] = []
    predecessors: list[tuple[int, ...]] = []

    def visit(row: int) -> None:
        if row == allowed_rows:
            predecessor = tuple(chosen)
            if predecessor != parts:
                predecessors.append(predecessor)
            return
        lower = parts[row + 1] if row + 1 < height else 0
        for width in range(lower, parts[row] + 1):
            chosen.append(width)
            visit(row + 1)
            chosen.pop()

    visit(0)
    return tuple(predecessors)


def _tableau_rows(
    parts: tuple[int, ...], max_entry: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate by successively removing the largest-entry horizontal strip."""
    if not parts:
        return ((),)

    output: list[tuple[tuple[int, ...], ...]] = []
    for largest_entry in range(len(parts), max_entry + 1):
        for predecessor in _horizontal_strip_predecessors(parts, largest_entry - 1):
            for smaller in _tableau_rows(predecessor, largest_entry - 1):
                rows = [list(row) for row in smaller]
                rows.extend([] for _ in range(len(parts) - len(rows)))
                for row, width in enumerate(parts):
                    previous_width = predecessor[row] if row < len(predecessor) else 0
                    rows[row].extend([largest_entry] * (width - previous_width))
                output.append(tuple(tuple(row) for row in rows))
    return tuple(sorted(output))


def enumerate_semistandard_young_tableaux(
    partition: IntegerPartition, max_entry: int
) -> SemistandardTableauEnumerationResult:
    """Return the complete lexicographically ordered bounded family."""
    count = semistandard_tableaux_count(partition, max_entry)
    size = sum(partition.parts)
    if count > MAX_SEMISTANDARD_TABLEAUX:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="semistandard_tableaux.count_bound",
            message=(
                "the complete semistandard-tableau family exceeds the admitted "
                f"count of {MAX_SEMISTANDARD_TABLEAUX}"
            ),
        )
    if count * size > MAX_ENUMERATED_CELLS:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="semistandard_tableaux.cell_bound",
            message=(
                "the complete semistandard-tableau family exceeds the admitted "
                f"aggregate cell count of {MAX_ENUMERATED_CELLS}"
            ),
        )
    # Every recursive edge removes at least one cell, and every emitted
    # predecessor shape has a completion. The search tree therefore has at
    # most count * size nodes. Generating an interlacing predecessor uses at
    # most size partial coordinates per child; sorting costs at most
    # count * log2(count) * size. This bound covers the actual search, copies,
    # and final ordering without charging impossible label prefixes.
    hook_content_work = size * max(1, partition.parts[0] if partition.parts else 0)
    work = hook_content_work + count * size * (size + (count - 1).bit_length() + 2)
    if work > MAX_ENUMERATION_WORK:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="semistandard_tableaux.work_bound",
            message="the complete family exceeds the admitted construction work",
        )
    estimated_bytes = count * (10 * size + 64) + 256
    if estimated_bytes > MAX_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="semistandard_tableaux.output_bound",
            message="the complete family exceeds the admitted result byte bound",
        )

    rows = _tableau_rows(partition.parts, max_entry) if count else ()
    tableaux = tuple(SemistandardYoungTableau(rows=value) for value in rows)
    if len(tableaux) != count:
        raise AssertionError("enumeration must agree with the hook-content count")
    return SemistandardTableauEnumerationResult(
        partition=partition,
        max_entry=max_entry,
        tableaux=tableaux,
    )


__all__ = [
    "enumerate_semistandard_young_tableaux",
    "semistandard_tableaux_count",
]
