"""Bounded exact Littlewood--Richardson coefficient computation."""

from __future__ import annotations

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.symmetric_functions._models import (
    MAX_LR_SEARCH_STATES,
    LittlewoodRichardsonCoefficientRequest,
    LittlewoodRichardsonCoefficientResult,
    SchurProductRequest,
    SchurProductResult,
    SchurProductTerm,
    _lr_inner_content_orientation,
)
from jacobian.math.combinatorics.symmetric_functions.values import IntegerPartition


def littlewood_richardson_coefficient(
    outer: IntegerPartition,
    inner: IntegerPartition,
    content: IntegerPartition,
) -> LittlewoodRichardsonCoefficientResult:
    """Return ``c^outer_{inner, content}`` under the fixed LR convention.

    An LR tableau is semistandard on ``outer / inner`` and has the submitted
    ``content``. Its reading word takes each row right-to-left, top-to-bottom;
    every prefix is lattice: ``count(i) >= count(i+1)`` for every ``i``.
    The search enumerates distinct multiset-word prefixes directly, with a
    complete precomputed upper bound based on the content multinomial.
    """
    request = LittlewoodRichardsonCoefficientRequest.model_validate(
        {
            "outer": outer.model_dump(mode="python"),
            "inner": inner.model_dump(mode="python"),
            "content": content.model_dump(mode="python"),
        }
    )
    return _compute_validated_lr(request)


def _compute_validated_lr(
    request: LittlewoodRichardsonCoefficientRequest,
) -> LittlewoodRichardsonCoefficientResult:
    """Run the LR kernel on a request whose bounds were already admitted.

    Both the native callable and the catalog wrapper share this one
    post-admission path; neither replays request validation.
    """
    coefficient = _lr_coefficient(request.outer, request.inner, request.content)
    return LittlewoodRichardsonCoefficientResult(
        outer=request.outer,
        inner=request.inner,
        content=request.content,
        coefficient=coefficient,
    )


def _lr_coefficient(
    outer: IntegerPartition,
    inner: IntegerPartition,
    content: IntegerPartition,
) -> int:
    """Count LR tableaux for partitions already inside the admitted envelope."""
    outer_parts = outer.parts
    inner_parts = inner.parts
    content_parts = content.parts

    if any(
        inner_parts[index] > (outer_parts[index] if index < len(outer_parts) else 0)
        for index in range(len(inner_parts))
    ):
        return 0

    skew_size = sum(outer_parts) - sum(inner_parts)
    if skew_size != sum(content_parts):
        return 0

    cells = tuple(
        (row, column)
        for row, outer_width in enumerate(outer_parts)
        for column in range(
            outer_width, inner_parts[row] if row < len(inner_parts) else 0, -1
        )
    )
    remaining = list(content_parts)
    assigned: dict[tuple[int, int], int] = {}
    prefix_counts = [0] * len(content_parts)
    visited = 0
    coefficient = 0

    def search(position: int) -> None:
        nonlocal visited, coefficient
        visited += 1
        if visited > MAX_LR_SEARCH_STATES:
            raise OperationResourceAdmissionError(
                location=("content",),
                code="symmetric_function.lr_search_states_exceeded",
                message="LR tableau search exceeded its admitted prefix bound",
            )
        if visited & 1023 == 0:
            request_checkpoint("during Littlewood-Richardson tableau search")
        if position == len(cells):
            coefficient += 1
            return

        row, column = cells[position]
        previous = cells[position - 1] if position else None
        for entry_index, available in enumerate(remaining):
            if available == 0:
                continue
            entry = entry_index + 1
            # Within a row the reading scan runs right-to-left, so entries
            # must weakly decrease along this scan.
            if (
                previous is not None
                and previous[0] == row
                and assigned[previous] < entry
            ):
                continue
            # All cells in upper rows have already been visited. Compare every
            # existing cell in this column, even when the skew shape has a gap.
            if any(
                upper_row < row
                and (upper_row, column) in assigned
                and assigned[(upper_row, column)] >= entry
                for upper_row in range(row)
            ):
                continue
            # The lattice condition is checked before appending this reading
            # letter, so it holds at every prefix, not only at the end.
            if entry > 1 and prefix_counts[entry - 2] <= prefix_counts[entry - 1]:
                continue

            remaining[entry_index] -= 1
            prefix_counts[entry_index] += 1
            assigned[(row, column)] = entry
            search(position + 1)
            del assigned[(row, column)]
            prefix_counts[entry_index] -= 1
            remaining[entry_index] += 1

    search(0)
    return coefficient


def schur_product(
    left: IntegerPartition, right: IntegerPartition
) -> SchurProductResult:
    """Return the complete bounded Schur expansion of ``s_left * s_right``."""
    request = SchurProductRequest.model_validate(
        {
            "left": left.model_dump(mode="python"),
            "right": right.model_dump(mode="python"),
        }
    )
    return _schur_product_from_request(request)


def _schur_product_from_request(request: SchurProductRequest) -> SchurProductResult:
    """Run complete expansion after the request has passed admission.

    Admission charged the cheaper commutative orientation once per candidate;
    the expansion reuses that orientation through the shared LR kernel on
    canonical partitions instead of constructing per-candidate requests.
    """
    degree = sum(request.left.parts) + sum(request.right.parts)
    inner, content = _lr_inner_content_orientation(request.left, request.right)
    terms = []
    for parts in _partitions_of(degree):
        outer = IntegerPartition(parts=parts)
        coefficient = _lr_coefficient(outer, inner, content)
        if coefficient:
            terms.append(SchurProductTerm(partition=outer, coefficient=coefficient))
    return SchurProductResult(
        left=request.left,
        right=request.right,
        terms=tuple(terms),
    )


def _partitions_of(
    total: int, maximum: int | None = None
) -> tuple[tuple[int, ...], ...]:
    """Enumerate partitions in descending lexicographic canonical order."""
    if total == 0:
        return ((),)
    limit = total if maximum is None else min(total, maximum)
    return tuple(
        (first, *rest)
        for first in range(limit, 0, -1)
        for rest in _partitions_of(total - first, first)
    )


__all__ = ["littlewood_richardson_coefficient", "schur_product"]
