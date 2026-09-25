"""Bounded exact count of semistandard tableaux with fixed content."""

from __future__ import annotations

from math import comb

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.algebraic.operations import (
    standard_young_tableaux_count,
)
from jacobian.math.combinatorics.semistandard_tableaux._models import (
    MAX_KOSTKA_RESULT_BYTES,
    MAX_KOSTKA_SEARCH_WORK,
    FixedContentCountRequest,
    FixedContentCountResult,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
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


def _result_byte_upper_bound(
    partition: IntegerPartition, content: TableauContent
) -> int:
    """Bound JSON bytes from canonical axes and the 500-cell scalar envelope."""

    size = sum(partition.parts)
    # Every valid count is at most size!, which is at most size**size.
    # This bounds decimal digits without constructing a large integer.
    count_digits = max(1, size * len(str(max(1, size))))
    # A sparse term contains a JSON-safe 16-digit label and a multiplicity no
    # larger than 500; 64 bytes per term covers keys, punctuation, and digits.
    return 512 + 4 * len(partition.parts) + 64 * len(content.terms) + count_digits


def _zero_result(
    partition: IntegerPartition, content: TableauContent
) -> FixedContentCountResult:
    return FixedContentCountResult(partition=partition, content=content, count=0)


def fixed_content_count(
    request: FixedContentCountRequest,
) -> FixedContentCountResult:
    """Return the number of SSYTs of ``partition`` with exact ``content``.

    Admission charges the complete multiset-word prefix tree before the
    row-major search. Each prefix scans at most the number of distinct source
    labels and performs at most two adjacent-cell comparisons. Special cases
    with closed forms avoid expanding that tree.
    """

    partition = request.partition
    content = request.content
    if _result_byte_upper_bound(partition, content) > MAX_KOSTKA_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("content",),
            code="semistandard_tableaux.kostka_output_bound",
            message="fixed-content result exceeds its admitted byte envelope",
        )

    request_checkpoint("before fixed-content tableau admission")
    size = sum(partition.parts)
    if content.size != size:
        return _zero_result(partition, content)
    if size == 0:
        return FixedContentCountResult(partition=partition, content=content, count=1)
    if len(partition.parts) == 1:
        # A weakly increasing row has exactly one filling for every fixed
        # multiset: its labels in increasing order.
        return FixedContentCountResult(partition=partition, content=content, count=1)
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
