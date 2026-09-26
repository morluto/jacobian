"""Pointwise addition of natural finite cellular-sheaf morphisms."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cellular_sheaves._kernel import _admit_field
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_ENTRY_DIGITS,
    MAX_SHEAF_MORPHISM_OUTPUT_CHARS,
    MAX_SHEAF_MORPHISM_WORK,
    FiniteCellularSheaf,
    sheaf_scalar_digits,
    sheaf_scalar_json_bound,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    SheafMorphismResult,
    morphism,
)


class SheafMorphismAddRequest(StrictModel):
    """Two natural maps with the same exact source and target sheaves."""

    left: SheafMorphismResult
    right: SheafMorphismResult


def add_morphisms(
    left: SheafMorphismResult, right: SheafMorphismResult
) -> SheafMorphismResult:
    """Return the pointwise sum of two natural maps with common parents.

    Each serialized naturality claim is re-established before use. Naturality
    is preserved by addition, so the output's property follows algebraically
    from the two admitted inputs and does not need a third diagram replay.
    """
    if type(left) is not SheafMorphismResult or type(right) is not SheafMorphismResult:
        raise OperationDomainValidationError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.value_type",
            message="morphism addition requires two cellular-sheaf morphism values",
        )
    if any(
        type(parent) is not FiniteCellularSheaf
        for parent in (left.source, left.target, right.source, right.target)
    ):
        raise OperationDomainValidationError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.parent_type",
            message="morphism parents must be checked finite cellular sheaves",
        )
    checked_left = morphism(left.source, left.target, left.components)
    checked_right = morphism(right.source, right.target, right.components)
    if not checked_left.natural or not checked_right.natural:
        raise OperationDomainValidationError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.non_natural",
            message="morphism addition requires two natural cellular-sheaf maps",
        )
    if (
        checked_left.source != checked_right.source
        or checked_left.target != checked_right.target
    ):
        raise OperationDomainValidationError(
            location=("right",),
            code="topology.cellular_sheaf.morphism_add.parent_mismatch",
            message="both maps must have exactly equal source and target sheaves",
        )

    source, target = checked_left.source, checked_left.target
    field = _admit_field(source.coefficient_field, source.prime)
    left_by_face = dict(checked_left.components)
    right_by_face = dict(checked_right.components)
    axis = source.canonical_face_order
    source_ranks = {stalk.simplex: len(stalk.basis) for stalk in source.stalks}
    target_ranks = {stalk.simplex: len(stalk.basis) for stalk in target.stalks}
    output_cells = sum(source_ranks[face] * target_ranks[face] for face in axis)
    work = output_cells
    if work > MAX_SHEAF_MORPHISM_WORK:
        raise OperationResourceAdmissionError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.work_bound",
            message="componentwise morphism addition exceeds its work bound",
        )
    # Each admitted input scalar has at most 64 decimal digits. Adding two
    # rationals needs at most two 64-digit cross-products and one carry digit.
    # The input morphism admission caps this output at 32,768 scalar cells, so
    # reserving 256 bytes per exact sum bounds the complete transient matrix at
    # 8,388,608 bytes before the values are constructed.

    # Both parent diagrams are retained in the returned morphism. Their exact
    # canonical wire sizes and the maximum composable component size are known
    # before any summed matrix is allocated.
    parent_chars = len(source.model_dump_json()) + len(target.model_dump_json())
    axis_chars = sum(64 + sum(2 + len(vertex) for vertex in face) for face in axis)
    admitted_output_chars = (
        parent_chars
        + axis_chars
        + sheaf_scalar_json_bound(output_cells, MAX_SHEAF_ENTRY_DIGITS)
        + 512
    )
    if admitted_output_chars > MAX_SHEAF_MORPHISM_OUTPUT_CHARS:
        raise OperationResourceAdmissionError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.output_bound",
            message="parent sheaves and summed components exceed the output bound",
        )

    exact_sums = []
    try:
        for face in axis:
            left_matrix = left_by_face[face]
            right_matrix = right_by_face[face]
            summed_values = tuple(
                tuple(
                    field.parse(a) + field.parse(b)
                    for a, b in zip(left_row, right_row, strict=True)
                )
                for left_row, right_row in zip(left_matrix, right_matrix, strict=True)
            )
            exact_sums.append((face, summed_values))
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise OperationDomainValidationError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.invalid_components",
            message="morphism components must be complete, rectangular exact matrices",
        ) from error

    max_output_digits = max(
        (
            sheaf_scalar_digits(field.typed(value))
            for _face, matrix in exact_sums
            for row in matrix
            for value in row
        ),
        default=1,
    )
    if max_output_digits > MAX_SHEAF_ENTRY_DIGITS:
        raise OperationResourceAdmissionError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.scalar_growth_bound",
            message="summed components exceed the scalar envelope used by morphism consumers",
        )
    components = tuple(
        (
            face,
            tuple(tuple(field.typed(value) for value in row) for row in matrix),
        )
        for face, matrix in exact_sums
    )

    return SheafMorphismResult(
        source=source,
        target=target,
        components=components,
        natural=True,
        obstruction=None,
    )


__all__ = ["SheafMorphismAddRequest", "add_morphisms"]
