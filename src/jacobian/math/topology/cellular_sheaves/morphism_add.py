"""Pointwise addition of natural finite cellular-sheaf morphisms."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cellular_sheaves._kernel import _admit_field
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_COVER_MAPS,
    MAX_SHEAF_DERIVED_RESTRICTIONS,
    MAX_SHEAF_ENTRY_DIGITS,
    MAX_SHEAF_MORPHISM_COMPONENT_CELLS,
    MAX_SHEAF_MORPHISM_OUTPUT_DIGIT_WORK,
    MAX_SHEAF_MORPHISM_PARENT_CELLS,
    MAX_SHEAF_MORPHISM_WORK,
    MAX_SHEAF_SIMPLICES,
    MAX_SHEAF_STALK_RANK,
    FiniteCellularSheaf,
    SheafRestriction,
    SheafStalk,
    sheaf_scalar_digit_work,
    sheaf_scalar_digits,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    SheafMorphismResult,
    morphism,
)
from jacobian.math.topology.cellular_sheaves.morphism_kernel import (
    _parent_reconstruction_bounds,
)


class SheafMorphismAddRequest(StrictModel):
    """Two natural maps with the same exact source and target sheaves."""

    left: SheafMorphismResult
    right: SheafMorphismResult


def _combined_work_bound(
    source: FiniteCellularSheaf,
    target: FiniteCellularSheaf,
    maps: tuple[SheafMorphismResult, SheafMorphismResult],
) -> int:
    axis = source.canonical_face_order
    if len(axis) > MAX_SHEAF_SIMPLICES:
        raise OperationResourceAdmissionError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.simplex_bound",
            message="morphism addition exceeds the admitted simplex bound",
        )
    if any(
        not isinstance(stalks, tuple)
        or len(stalks) > MAX_SHEAF_SIMPLICES
        or any(type(stalk) is not SheafStalk for stalk in stalks)
        for stalks in (source.stalks, target.stalks)
    ):
        raise OperationDomainValidationError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.stalk_structure",
            message="morphism parents must contain bounded canonical stalk values",
        )
    source_stalks = {stalk.simplex: stalk for stalk in source.stalks}
    target_stalks = {stalk.simplex: stalk for stalk in target.stalks}
    if tuple(source_stalks) != axis or tuple(target_stalks) != axis:
        raise OperationDomainValidationError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.stalk_axis",
            message="morphism parents must bind stalks to canonical simplex axes",
        )
    if any(
        len(stalk.basis) > MAX_SHEAF_STALK_RANK
        for stalk in (*source.stalks, *target.stalks)
    ):
        raise OperationResourceAdmissionError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.stalk_rank_bound",
            message="morphism addition exceeds the admitted stalk-rank bound",
        )
    for sheaf in (source, target):
        if (
            not isinstance(sheaf.cover_restrictions, tuple)
            or len(sheaf.cover_restrictions) > MAX_SHEAF_COVER_MAPS
            or not isinstance(sheaf.derived_restrictions, tuple)
            or len(sheaf.derived_restrictions) > MAX_SHEAF_DERIVED_RESTRICTIONS
        ):
            raise OperationResourceAdmissionError(
                location=(),
                code="topology.cellular_sheaf.morphism_add.cover_map_bound",
                message="morphism addition exceeds the admitted restriction-map bound",
            )
    parent_scalar_cells = _parent_scalar_cells((source, target))
    component_cells = _component_input_cells(maps)

    square_work = sum(
        len(target_stalks[restriction.target].basis)
        * len(source_stalks[restriction.source].basis)
        * (
            len(target_stalks[restriction.source].basis)
            + len(source_stalks[restriction.target].basis)
        )
        for restriction in source.cover_restrictions
    )
    reconstruction_work, _ = _parent_reconstruction_bounds(
        (source, target), input_digits=MAX_SHEAF_ENTRY_DIGITS
    )
    addition_cells = sum(
        len(source_stalks[cell].basis) * len(target_stalks[cell].basis) for cell in axis
    )
    # Two input morphisms each reconstruct both parents and check every cover
    # square. The two full diagram checks and the pointwise sum also count
    # against this operation's one work budget.
    combined_work = (
        2 * square_work
        + 2 * reconstruction_work
        + 4 * len(axis) ** 2
        + component_cells
        + 2 * parent_scalar_cells
        + addition_cells
    )
    if combined_work > MAX_SHEAF_MORPHISM_WORK:
        raise OperationResourceAdmissionError(
            location=(),
            code="topology.cellular_sheaf.morphism_add.work_bound",
            message="input admission and componentwise addition exceed the combined work bound",
        )
    return addition_cells


def _component_input_cells(
    maps: tuple[SheafMorphismResult, SheafMorphismResult],
) -> int:
    total = 0
    for morphism_value in maps:
        components = morphism_value.components
        if not isinstance(components, tuple) or len(components) > MAX_SHEAF_SIMPLICES:
            raise OperationDomainValidationError(
                location=(),
                code="topology.cellular_sheaf.morphism_add.component_structure",
                message="morphism components must be bounded simplex-matrix pairs",
            )
        for component in components:
            if (
                not isinstance(component, tuple)
                or len(component) != 2
                or not isinstance(component[1], (tuple, list))
                or len(component[1]) > MAX_SHEAF_STALK_RANK
                or any(
                    not isinstance(row, (tuple, list))
                    or len(row) > MAX_SHEAF_STALK_RANK
                    for row in component[1]
                )
            ):
                raise OperationDomainValidationError(
                    location=(),
                    code="topology.cellular_sheaf.morphism_add.component_structure",
                    message="morphism components must be bounded simplex-matrix pairs",
                )
            total += sum(len(row) for row in component[1])
        if total > 2 * MAX_SHEAF_MORPHISM_COMPONENT_CELLS:
            raise OperationResourceAdmissionError(
                location=(),
                code="topology.cellular_sheaf.morphism_add.component_cells_bound",
                message="input component matrices exceed their combined cell bound",
            )
    return total


def _parent_scalar_cells(
    parents: tuple[FiniteCellularSheaf, FiniteCellularSheaf],
) -> int:
    total = 0
    for sheaf in parents:
        for restriction in (*sheaf.cover_restrictions, *sheaf.derived_restrictions):
            if (
                type(restriction) is not SheafRestriction
                or not isinstance(restriction.entries, tuple)
                or len(restriction.entries) > MAX_SHEAF_STALK_RANK
                or any(
                    not isinstance(row, tuple) or len(row) > MAX_SHEAF_STALK_RANK
                    for row in restriction.entries
                )
            ):
                raise OperationDomainValidationError(
                    location=(),
                    code="topology.cellular_sheaf.morphism_add.restriction_structure",
                    message="morphism parent restrictions must have bounded matrices",
                )
            total += sum(len(row) for row in restriction.entries)
    return total


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
    if left.source != right.source or left.target != right.target:
        raise OperationDomainValidationError(
            location=("right",),
            code="topology.cellular_sheaf.morphism_add.parent_mismatch",
            message="both maps must have exactly equal source and target sheaves",
        )
    source, target = left.source, left.target
    output_cells = _combined_work_bound(source, target, (left, right))
    checked_left = morphism(source, target, left.components)
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
    # Each admitted input scalar has at most MAX_SHEAF_ENTRY_DIGITS decimal
    # digits. Adding two rationals needs at most two such cross-products and
    # one carry digit, so the summed matrix is bounded by exact scalar
    # allocation and digit work rather than by an estimated encoded size.
    #
    # Both parent diagrams are retained in the returned morphism. Their face
    # axes and the maximum composable component size are known before any
    # summed matrix is allocated.
    axis_cells = sum(
        MAX_SHEAF_ENTRY_DIGITS + sum(2 + len(vertex) for vertex in face)
        for face in axis
    )
    admitted_output_digit_work = (
        2 * MAX_SHEAF_MORPHISM_PARENT_CELLS
        + axis_cells
        + sheaf_scalar_digit_work(output_cells, 2 * MAX_SHEAF_ENTRY_DIGITS + 1)
    )
    if admitted_output_digit_work > MAX_SHEAF_MORPHISM_OUTPUT_DIGIT_WORK:
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
