"""Bounded exact Littlewood--Richardson coefficient computation."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, ValidationError

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.symmetric_functions._models import (
    MAX_LR_SEARCH_STATES,
    MAX_LR_SKEW_CELLS,
    MAX_LR_TABLEAU_OUTPUT_BYTES,
    MAX_LR_TABLEAUX,
    LittlewoodRichardsonCoefficientRequest,
    LittlewoodRichardsonCoefficientResult,
    LittlewoodRichardsonTableauxRequest,
    LittlewoodRichardsonTableauxResult,
    SchurProductRequest,
    SchurProductResult,
    SchurProductTerm,
    _lr_complete_word_bound,
    _lr_inner_content_orientation,
    _lr_prefix_state_bound,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    TableauCandidate,
)


def _admit_native_request[RequestT: BaseModel](
    model: type[RequestT], values: dict[str, object]
) -> RequestT:
    try:
        if any(type(value) is not IntegerPartition for value in values.values()):
            raise TypeError("native LR arguments must be IntegerPartition values")
        native_values = {
            key: value.model_dump(mode="python") for key, value in values.items()
        }
        request = model.model_validate(native_values)
    except (ValidationError, AttributeError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=(),
            code="symmetric_functions.littlewood_richardson.invalid_request",
            message="native LR arguments must be canonical partitions within the operation envelope",
        ) from exc
    if isinstance(
        request,
        (LittlewoodRichardsonCoefficientRequest, LittlewoodRichardsonTableauxRequest),
    ):
        _admit_lr_resources(request)
    return request


def _admit_lr_resources(
    request: LittlewoodRichardsonCoefficientRequest
    | LittlewoodRichardsonTableauxRequest,
) -> None:
    skew_size = sum(request.outer.parts) - sum(request.inner.parts)
    content_size = sum(request.content.parts)
    if isinstance(request, LittlewoodRichardsonTableauxRequest):
        inner_contained = all(
            part
            <= (request.outer.parts[index] if index < len(request.outer.parts) else 0)
            for index, part in enumerate(request.inner.parts)
        )
        if not inner_contained or skew_size != content_size:
            return
    if skew_size > MAX_LR_SKEW_CELLS:
        raise OperationResourceAdmissionError(
            location=("outer",),
            code="symmetric_functions.lr_skew_size_exceeded",
            message=f"LR skew size must not exceed {MAX_LR_SKEW_CELLS}",
        )
    if content_size > MAX_LR_SKEW_CELLS:
        raise OperationResourceAdmissionError(
            location=("content",),
            code="symmetric_functions.lr_content_size_exceeded",
            message=f"LR content size must not exceed {MAX_LR_SKEW_CELLS}",
        )
    states = _lr_prefix_state_bound(request.content)
    if states > MAX_LR_SEARCH_STATES:
        raise OperationResourceAdmissionError(
            location=("content",),
            code="symmetric_functions.lr_search_states_exceeded",
            message=f"LR search prefix bound exceeds {MAX_LR_SEARCH_STATES}",
        )
    if isinstance(request, LittlewoodRichardsonTableauxRequest):
        complete_words = _lr_complete_word_bound(request.content)
        context_bytes = 1024 + 48 * (
            len(request.outer.parts)
            + len(request.inner.parts)
            + len(request.content.parts)
        )
        output_bytes = context_bytes + complete_words * (64 + 16 * MAX_LR_SKEW_CELLS)
        if output_bytes > MAX_LR_TABLEAU_OUTPUT_BYTES:
            raise OperationResourceAdmissionError(
                location=("content",),
                code="symmetric_functions.lr_tableau_output_exceeded",
                message="complete LR tableau family exceeds its output byte bound",
            )
        if complete_words > MAX_LR_TABLEAUX:
            raise OperationResourceAdmissionError(
                location=("content",),
                code="symmetric_functions.lr_tableau_count_exceeded",
                message=f"complete LR tableau family exceeds {MAX_LR_TABLEAUX} candidates",
            )


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
    request = _admit_native_request(
        LittlewoodRichardsonCoefficientRequest,
        {
            "outer": outer,
            "inner": inner,
            "content": content,
        },
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
    outer_parts, inner_parts, content_parts = outer.parts, inner.parts, content.parts
    if any(
        inner_parts[index] > (outer_parts[index] if index < len(outer_parts) else 0)
        for index in range(len(inner_parts))
    ):
        return 0
    if sum(outer_parts) - sum(inner_parts) != sum(content_parts):
        return 0
    return sum(
        1 for _ in _iter_lr_tableau_rows(outer_parts, inner_parts, content_parts)
    )


def _iter_lr_tableau_rows(
    outer_parts: tuple[int, ...],
    inner_parts: tuple[int, ...],
    content_parts: tuple[int, ...],
) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Yield LR fillings in reading-word order under the admitted envelope."""
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

    def search(position: int) -> Iterator[tuple[tuple[int, ...], ...]]:
        nonlocal visited
        visited += 1
        if visited > MAX_LR_SEARCH_STATES:
            raise OperationResourceAdmissionError(
                location=("content",),
                code="symmetric_function.lr_search_states_exceeded",
                message="LR tableau search exceeded its admitted prefix bound",
            )
        if visited & 1023 == 0:
            request_checkpoint("during Littlewood-Richardson tableau enumeration")
        if position == len(cells):
            present_rows = tuple(
                row
                for row, width in enumerate(outer_parts)
                if width > (inner_parts[row] if row < len(inner_parts) else 0)
            )
            yield tuple(
                tuple(
                    assigned[(row, column)]
                    for column in range(
                        (inner_parts[row] if row < len(inner_parts) else 0) + 1,
                        outer_parts[row] + 1,
                    )
                )
                for row in present_rows
            )
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
            yield from search(position + 1)
            del assigned[(row, column)]
            prefix_counts[entry_index] -= 1
            remaining[entry_index] += 1

    yield from search(0)


def littlewood_richardson_tableaux(
    outer: IntegerPartition,
    inner: IntegerPartition,
    content: IntegerPartition,
) -> LittlewoodRichardsonTableauxResult:
    """Enumerate the complete LR tableau family in canonical reading-word order.

    The operation uses the same bounded prefix envelope as coefficient
    computation, plus an admission bound for worst-case result count and
    serialized growth. Every recursive path corresponds to one distinct
    content-prefix, and every complete path to exactly one skew filling.
    """
    request = _admit_native_request(
        LittlewoodRichardsonTableauxRequest,
        {"outer": outer, "inner": inner, "content": content},
    )
    return _enumerate_validated_lr(request)


def _enumerate_validated_lr(
    request: LittlewoodRichardsonTableauxRequest,
) -> LittlewoodRichardsonTableauxResult:
    outer, inner, content = request.outer, request.inner, request.content
    outer_parts, inner_parts = outer.parts, inner.parts
    if any(
        inner_parts[index] > (outer_parts[index] if index < len(outer_parts) else 0)
        for index in range(len(inner_parts))
    ):
        return LittlewoodRichardsonTableauxResult._from_kernel(request, ())
    skew_size = sum(outer_parts) - sum(inner_parts)
    if skew_size != sum(content.parts):
        return LittlewoodRichardsonTableauxResult._from_kernel(request, ())

    # Right-to-left, top-to-bottom is the declared LR reading word. The shared
    # lazy kernel makes coefficient and complete-family outputs use identical
    # row, column, content, and lattice constraints without materializing the
    # family for coefficient requests.
    tableaux = tuple(
        TableauCandidate(rows=rows)
        for rows in _iter_lr_tableau_rows(outer_parts, inner_parts, content.parts)
    )
    return LittlewoodRichardsonTableauxResult._from_kernel(request, tableaux)


def schur_product(
    left: IntegerPartition, right: IntegerPartition
) -> SchurProductResult:
    """Return the complete bounded Schur expansion of ``s_left * s_right``."""
    request = _admit_native_request(
        SchurProductRequest,
        {
            "left": left,
            "right": right,
        },
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


__all__ = [
    "littlewood_richardson_coefficient",
    "littlewood_richardson_tableaux",
    "schur_product",
]
