"""Bounded exact count of semistandard tableaux with fixed content."""

from __future__ import annotations

from math import comb

from pydantic import ValidationError

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.algebraic.operations import (
    standard_young_tableaux_count,
)
from jacobian.math.combinatorics.semistandard_tableaux._models import (
    MAX_KOSTKA_RESULT_SIZE,
    MAX_KOSTKA_SEARCH_WORK,
    FixedContentCountRequest,
    FixedContentCountResult,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    MAX_PARTITION_SIZE,
    IntegerPartition,
    TableauContent,
)


def _multiset_word_count(content: TableauContent) -> int:
    """Count all distinct words with the source content, by multinomials."""

    ways = 1
    placed = 0
    for term in content.terms:
        request_checkpoint("during Kostka multinomial preflight")
        multiplicity = term.multiplicity
        ways *= comb(placed + multiplicity, multiplicity)
        placed += multiplicity
    return ways


def _count_by_row_major_search(
    partition: IntegerPartition, content: TableauContent
) -> int:
    """Count feasible multiset words by checking adjacent tableau cells."""

    row_starts: list[int] = []
    left_cells: list[int] = []
    above_cells: list[int] = []
    position = 0
    for row, width in enumerate(partition.parts):
        row_starts.append(position)
        for column in range(width):
            left_cells.append(position + column - 1 if column else -1)
            above_cells.append(row_starts[row - 1] + column if row else -1)
        position += width

    entries = tuple(term.entry for term in content.terms)
    remaining = [term.multiplicity for term in content.terms]
    values = [0] * position
    total = 0

    def visit(cell: int) -> None:
        nonlocal total
        request_checkpoint("during fixed-content tableau count")
        if cell == position:
            total += 1
            return

        left = left_cells[cell]
        above = above_cells[cell]
        for index, entry in enumerate(entries):
            if remaining[index] == 0:
                continue
            if left >= 0 and values[left] > entry:
                continue
            if above >= 0 and values[above] >= entry:
                continue
            remaining[index] -= 1
            values[cell] = entry
            visit(cell + 1)
            remaining[index] += 1

    visit(0)
    return total


def _result_size_upper_bound(
    partition: IntegerPartition, content: TableauContent
) -> int:
    """Bound the retained result from canonical axes and the 500-cell scalar envelope.

    The units are scalar count times maximum decimal width plus a fixed
    overhead allowance, not an encoded transport measurement.
    """

    size = sum(partition.parts)
    # Every valid count is at most size!, which is at most size**size.
    # This bounds decimal digits without constructing a large integer.
    count_digits = max(1, size * len(str(max(1, size))))
    # A sparse term contains a JSON-safe 16-digit label and a multiplicity no
    # larger than 500; 64 bytes per term covers keys, punctuation, and digits.
    return 512 + 4 * len(partition.parts) + 64 * len(content.terms) + count_digits


def _forced_full_height_columns_count(
    partition: IntegerPartition, content: TableauContent
) -> int | None:
    """Reduce forced full-height columns until one row or a general case remains."""

    rows = list(partition.parts)
    counts = [term.multiplicity for term in content.terms]
    row_minima = [0] * len(rows)
    while rows:
        active_labels = [index for index, count in enumerate(counts) if count]
        if len(rows) == 1:
            if sum(counts) == rows[0] and all(
                index >= row_minima[0] for index in active_labels
            ):
                return 1
            return 0
        if len(active_labels) < len(rows):
            return 0
        if len(active_labels) > len(rows):
            return None

        width = rows[-1]
        if any(counts[index] < width for index in active_labels):
            return 0
        if any(
            label_index < row_minima[row_index]
            for row_index, label_index in enumerate(active_labels)
        ):
            return 0

        for row_index, label_index in enumerate(active_labels):
            counts[label_index] -= width
            rows[row_index] -= width
            row_minima[row_index] = label_index
        while rows and rows[-1] == 0:
            rows.pop()
            row_minima.pop()
    return 1 if not any(counts) else 0


def _count_two_row_content(partition: IntegerPartition, content: TableauContent) -> int:
    """Count two-row tableaux by bounded row-content dynamic programming."""

    top_width = partition.parts[0]
    current = [0] * (top_width + 1)
    current[0] = 1
    prefix_content = 0
    visited_states = 0
    for term in content.terms:
        prefix_content += term.multiplicity
        differences = [0] * (top_width + 2)
        for top_before, count in enumerate(current):
            visited_states += 1
            if visited_states % 256 == 0:
                request_checkpoint("during two-row Kostka reduction")
            if count == 0:
                continue
            first_top_after = max(top_before, prefix_content - top_before)
            last_top_after = min(top_width, top_before + term.multiplicity)
            if first_top_after <= last_top_after:
                differences[first_top_after] += count
                differences[last_top_after + 1] -= count
        running = 0
        next_counts = []
        for delta in differences[:-1]:
            running += delta
            next_counts.append(running)
        current = next_counts
    return current[top_width]


def _two_row_count_result(
    partition: IntegerPartition, content: TableauContent
) -> FixedContentCountResult | None:
    if len(partition.parts) != 2:
        return None
    work_bound = len(content.terms) * (partition.parts[0] + 1)
    if work_bound > MAX_KOSTKA_SEARCH_WORK:
        raise OperationResourceAdmissionError(
            location=("content",),
            code="semistandard_tableaux.kostka_search_bound",
            message=(
                "two-row fixed-content reduction exceeds its admitted "
                f"work bound of {MAX_KOSTKA_SEARCH_WORK} units"
            ),
        )
    return FixedContentCountResult(
        partition=partition,
        content=content,
        count=_count_two_row_content(partition, content),
    )


def _zero_result(
    partition: IntegerPartition, content: TableauContent
) -> FixedContentCountResult:
    return FixedContentCountResult(partition=partition, content=content, count=0)


def fixed_content_count(
    partition: IntegerPartition | FixedContentCountRequest,
    content: TableauContent | None = None,
) -> FixedContentCountResult:
    """Return the number of SSYTs of ``partition`` with exact ``content``.

    Admission charges the complete multiset-word prefix tree before the
    row-major search. Each prefix scans at most the number of distinct source
    labels and performs at most two adjacent-cell comparisons. Special cases
    with closed forms avoid expanding that tree.
    """

    if isinstance(partition, FixedContentCountRequest):
        if content is not None:
            raise TypeError("content must be omitted when passing a count request")
        content = partition.content
        partition = partition.partition
    elif content is None:
        raise TypeError("content is required when passing a partition")
    # Check exact carrier and container shapes before dumping or traversing a
    # potentially forged model_construct() instance. These length checks are O(1).
    if (
        type(partition) is not IntegerPartition
        or type(partition.parts) is not tuple
        or len(partition.parts) > MAX_PARTITION_SIZE
        or type(content) is not TableauContent
        or type(content.terms) is not tuple
        or len(content.terms) > MAX_PARTITION_SIZE
    ):
        raise OperationDomainValidationError(
            location=(),
            code="semistandard_tableaux.fixed_content_input_invalid",
            message="partition and content must be canonical bounded values",
        )
    try:
        partition = IntegerPartition.model_validate(partition.model_dump())
        content = TableauContent.model_validate(content.model_dump())
    except (AttributeError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=(),
            code="semistandard_tableaux.fixed_content_input_invalid",
            message="partition and content must be canonical bounded values",
        ) from exc
    if _result_size_upper_bound(partition, content) > MAX_KOSTKA_RESULT_SIZE:
        raise OperationResourceAdmissionError(
            location=("content",),
            code="semistandard_tableaux.kostka_output_bound",
            message="fixed-content result exceeds its admitted representation envelope",
        )

    request_checkpoint("before fixed-content tableau admission")
    size = sum(partition.parts)
    if content.size != size:
        return _zero_result(partition, content)
    if size == 0 or len(partition.parts) == 1:
        # The empty tableau is unique; a weakly increasing row has exactly
        # one filling for every fixed multiset.
        return FixedContentCountResult(partition=partition, content=content, count=1)
    two_row_result = _two_row_count_result(partition, content)
    if two_row_result is not None:
        return two_row_result
    if (
        len(partition.parts) > 1
        and len(content.terms) == len(partition.parts)
        and tuple(term.multiplicity for term in content.terms) == partition.parts
    ):
        # K_{lambda,lambda}=1: each row is forced to its correspondingly
        # ordered label, and the partition inequalities make columns strict.
        return FixedContentCountResult(partition=partition, content=content, count=1)
    forced_count = _forced_full_height_columns_count(partition, content)
    if forced_count is not None:
        return FixedContentCountResult(
            partition=partition, content=content, count=forced_count
        )
    if len(content.terms) < len(partition.parts):
        # A strict column of this height needs at least this many distinct
        # labels, regardless of their multiplicities.
        return _zero_result(partition, content)
    if all(term.multiplicity == 1 for term in content.terms):
        request_checkpoint("before standard-tableau hook count")
        # Relabeling the sorted source labels by 1..n is a bijection to standard
        # tableaux; use the existing hook-length count instead of a search.
        count = standard_young_tableaux_count(partition)
        return FixedContentCountResult(
            partition=partition, content=content, count=count
        )
    words = _multiset_word_count(content)
    prefixes_upper = (size + 1) * words
    active_labels = len(content.terms)
    work_upper = prefixes_upper * (active_labels + 2)
    if work_upper > MAX_KOSTKA_SEARCH_WORK:
        raise OperationResourceAdmissionError(
            location=("content",),
            code="semistandard_tableaux.kostka_search_bound",
            message=(
                "the complete fixed-content multiset search exceeds its admitted "
                f"work bound of {MAX_KOSTKA_SEARCH_WORK} units"
            ),
        )

    request_checkpoint("before fixed-content tableau search")
    count = _count_by_row_major_search(partition, content)
    request_checkpoint("before fixed-content count result construction")
    return FixedContentCountResult(partition=partition, content=content, count=count)


__all__ = ["fixed_content_count"]
