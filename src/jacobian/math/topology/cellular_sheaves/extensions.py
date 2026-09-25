"""Sections, restriction queries, and natural morphisms of finite sheaves."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Any, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology.cellular_sheaves._kernel import (
    _admit_field,
    _cochain_nullspace,
    _cochain_rref,
    _ExactField,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_COVER_MAPS,
    MAX_SHEAF_DERIVED_RESTRICTIONS,
    MAX_SHEAF_ENTRY_DIGITS,
    MAX_SHEAF_MORPHISM_COMPONENT_CELLS,
    MAX_SHEAF_MORPHISM_OUTPUT_CHARS,
    MAX_SHEAF_MORPHISM_WORK,
    MAX_SHEAF_RESTRICTION_CELLS,
    MAX_SHEAF_SECTION_MATRIX_CELLS,
    MAX_SHEAF_SECTION_OUTPUT_CELLS,
    MAX_SHEAF_SECTION_OUTPUT_CHARS,
    MAX_SHEAF_SECTION_RESTRICTION_OUTPUT_CHARS,
    MAX_SHEAF_SECTION_WORK,
    MAX_SHEAF_SIMPLICES,
    MAX_SHEAF_STALK_RANK,
    MAX_SHEAF_TOTAL_STALK_RANK,
    FiniteCellularSheaf,
    SheafCochainCoordinate,
    SheafRestriction,
    SheafScalar,
    SheafSectionCompatibilityAxis,
    SheafSectionEvaluation,
    SheafSectionRestriction,
    SheafSectionSpace,
    SheafStalk,
    _require_field_scalars,
    sheaf_scalar_digits,
    sheaf_scalar_json_bound,
)
from jacobian.math.topology.cellular_sheaves.subcomplex import restrict_to_subcomplex


class SheafSectionsRequest(StrictModel):
    sheaf: FiniteCellularSheaf


class SheafSectionRestrictionRequest(StrictModel):
    sheaf: FiniteCellularSheaf
    subcomplex: FiniteSimplicialComplex


class SheafRestrictionRequest(StrictModel):
    sheaf: FiniteCellularSheaf
    source: tuple[str, ...]
    target: tuple[str, ...]


class SheafRestrictionResult(StrictModel):
    sheaf: FiniteCellularSheaf
    restriction: SheafRestriction


ComponentKey = str | tuple[str, ...]
Component = tuple[ComponentKey, tuple[tuple[SheafScalar, ...], ...]]


class SheafMorphismRequest(StrictModel):
    source: FiniteCellularSheaf
    target: FiniteCellularSheaf
    # Tuple keys are the unambiguous canonical simplex axis.  A dotted string
    # remains accepted for legacy, unambiguous axes only.
    components: tuple[Component, ...] = Field(
        default=(),
        description=(
            "Components in canonical stalk order; use simplex tuple keys when "
            "vertex labels contain dots, because dotted string keys are ambiguous."
        ),
    )


class SheafMorphismResult(StrictModel):
    source: FiniteCellularSheaf
    target: FiniteCellularSheaf
    components: tuple[Component, ...]
    natural: bool
    obstruction: str | None = None

    @model_validator(mode="after")
    def require_component_scalar_parent(self) -> Self:
        if (
            self.source.coefficient_field != self.target.coefficient_field
            or self.source.prime != self.target.prime
        ):
            raise ValueError("morphism parents must have the same exact field")
        _require_field_scalars(
            tuple(matrix for _, matrix in self.components),
            self.source.coefficient_field,
            self.source.prime,
            label="morphism component",
        )
        return self


class SheafMorphismComposeRequest(StrictModel):
    """Compose checked maps ``first: F -> G`` and ``second: G -> H``."""

    first: SheafMorphismResult
    second: SheafMorphismResult


def _section_resource(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("sheaf",),
        code=f"topology.cellular_sheaf.sections.{code}",
        message=message,
    )


def _section_domain(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("sheaf",),
        code=f"topology.cellular_sheaf.sections.{code}",
        message=message,
    )


@dataclass(frozen=True)
class _SectionPlan:
    sheaf: FiniteCellularSheaf
    field: _ExactField
    cells: tuple[tuple[str, ...], ...]
    stalk_for: dict[tuple[str, ...], SheafStalk]
    pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    restriction_for: dict[tuple[tuple[str, ...], tuple[str, ...]], SheafRestriction]
    ambient_dimension: int
    row_count: int
    matrix_cells: int


def _admit_section_plan(sheaf: FiniteCellularSheaf) -> _SectionPlan:
    field: _ExactField = _admit_field(sheaf.coefficient_field, sheaf.prime)
    cells = sheaf.canonical_face_order
    if len(cells) > MAX_SHEAF_SIMPLICES:
        raise _section_resource(
            "simplex_bound",
            f"global sections admit at most {MAX_SHEAF_SIMPLICES} simplices",
        )

    stalk_for = {stalk.simplex: stalk for stalk in sheaf.stalks}
    if len(stalk_for) != len(cells) or tuple(stalk_for) != cells:
        raise _section_domain(
            "stalk_axis", "the sheaf must bind exactly one stalk to every simplex"
        )
    ambient_dimension = 0
    for stalk in sheaf.stalks:
        if len(stalk.basis) > MAX_SHEAF_STALK_RANK:
            raise _section_resource(
                "stalk_rank_bound",
                f"a stalk exceeds the {MAX_SHEAF_STALK_RANK}-coordinate bound",
            )
        ambient_dimension += len(stalk.basis)
    if ambient_dimension > MAX_SHEAF_TOTAL_STALK_RANK:
        raise _section_resource(
            "ambient_dimension_bound",
            "the global section ambient stalk axis exceeds its coordinate bound",
        )

    pairs = tuple(
        (source, target)
        for source in cells
        for target in cells
        if set(source) < set(target)
    )
    restrictions = (*sheaf.cover_restrictions, *sheaf.derived_restrictions)
    if (
        len(sheaf.cover_restrictions) > MAX_SHEAF_COVER_MAPS
        or len(sheaf.derived_restrictions) > MAX_SHEAF_DERIVED_RESTRICTIONS
    ):
        raise _section_resource(
            "restriction_count_bound", "the sheaf restriction diagram exceeds its bound"
        )
    restriction_cells = sum(
        len(item.entries) * (len(item.column_basis) + len(item.row_basis))
        for item in restrictions
    )
    if restriction_cells > MAX_SHEAF_RESTRICTION_CELLS:
        raise _section_resource(
            "restriction_cells_bound",
            "the input restriction matrices exceed their cell bound",
        )
    restriction_for = {(item.source, item.target): item for item in restrictions}
    if len(restriction_for) != len(restrictions) or set(restriction_for) != set(pairs):
        raise _section_domain(
            "restriction_diagram_incomplete",
            "global sections require exactly one restriction for every canonical strict face inclusion",
        )

    row_count = sum(len(stalk_for[target].basis) for _, target in pairs)
    matrix_cells = row_count * ambient_dimension
    if matrix_cells > MAX_SHEAF_SECTION_MATRIX_CELLS:
        raise _section_resource(
            "compatibility_matrix_bound",
            f"the full compatibility matrix has {matrix_cells} cells, above "
            f"the {MAX_SHEAF_SECTION_MATRIX_CELLS}-cell bound",
        )
    work_bound = matrix_cells * min(row_count, ambient_dimension)
    if not row_count:
        work_bound = ambient_dimension * ambient_dimension
    if work_bound > MAX_SHEAF_SECTION_WORK:
        raise _section_resource(
            "work_bound",
            f"nullspace work bound {work_bound} exceeds {MAX_SHEAF_SECTION_WORK}",
        )
    maximum_output_cells = (
        matrix_cells + 2 * ambient_dimension * ambient_dimension + row_count
    )
    if maximum_output_cells > MAX_SHEAF_SECTION_OUTPUT_CELLS:
        raise _section_resource(
            "output_cells_bound",
            "the compatibility matrix and worst-case section basis exceed the output-cell bound",
        )
    return _SectionPlan(
        sheaf=sheaf,
        field=field,
        cells=cells,
        stalk_for=stalk_for,
        pairs=pairs,
        restriction_for=restriction_for,
        ambient_dimension=ambient_dimension,
        row_count=row_count,
        matrix_cells=matrix_cells,
    )


def _parse_section_restrictions(
    plan: _SectionPlan,
) -> tuple[
    dict[tuple[tuple[str, ...], tuple[str, ...]], tuple[tuple[object, ...], ...]],
    int,
    int,
    int,
]:
    parsed: dict[
        tuple[tuple[str, ...], tuple[str, ...]], tuple[tuple[object, ...], ...]
    ] = {}
    max_input_digits = 1
    max_input_chars = 1
    input_chars = 0
    for key, restriction in plan.restriction_for.items():
        matrix: list[tuple[object, ...]] = []
        for row in restriction.entries:
            parsed_row: list[object] = []
            for entry in row:
                digits = sheaf_scalar_digits(entry)
                if digits > MAX_SHEAF_ENTRY_DIGITS:
                    raise _section_resource(
                        "coefficient_digits_bound",
                        f"a restriction coefficient exceeds {MAX_SHEAF_ENTRY_DIGITS} decimal digits",
                    )
                max_input_digits = max(max_input_digits, digits)
                input_chars += sheaf_scalar_json_bound(1, digits)
                max_input_chars = max(max_input_chars, 2 * digits + 32)
                try:
                    parsed_row.append(plan.field.parse(entry))
                except (ValueError, ZeroDivisionError, OverflowError) as exc:
                    raise _section_domain(
                        "coefficient_invalid",
                        f"restriction entries must be valid scalars for the declared field: {exc}",
                    ) from exc
            matrix.append(tuple(parsed_row))
        parsed[key] = tuple(matrix)
    return parsed, input_chars, max_input_chars, max_input_digits


def _require_section_height_bound(
    plan: _SectionPlan, input_chars: int, max_input_chars: int, max_input_digits: int
) -> int:
    pivot_bound = min(plan.row_count, plan.ambient_dimension)
    if plan.sheaf.coefficient_field.value == "QQ":
        # Each equation row has at most one source-stalk block (rank <= 8)
        # and one target coordinate. Clearing those row denominators and
        # applying Hadamard's determinant bound bounds every RREF/nullspace
        # coordinate by the corresponding minors before exact elimination.
        max_result_scalar_chars = (
            2 * (MAX_SHEAF_STALK_RANK + 1) * max_input_digits + 2
        ) * pivot_bound + 1
    else:
        max_result_scalar_chars = len(str(plan.sheaf.prime))
    growth_output_chars = (
        input_chars
        + plan.matrix_cells * max_input_chars
        + 2 * plan.ambient_dimension * plan.ambient_dimension * max_result_scalar_chars
        + plan.row_count
        * (2 * max((len(".".join(cell)) for cell in plan.cells), default=1) + 64)
    )
    if growth_output_chars > MAX_SHEAF_SECTION_OUTPUT_CHARS:
        raise _section_resource(
            "intermediate_height_bound",
            "the determinant-based exact output bound exceeds the section output envelope",
        )
    return max_result_scalar_chars


def _section_matrix_and_axes(
    plan: _SectionPlan,
    parsed: dict[
        tuple[tuple[str, ...], tuple[str, ...]], tuple[tuple[object, ...], ...]
    ],
) -> tuple[
    tuple[SheafCochainCoordinate, ...],
    list[list[object]],
    tuple[SheafSectionCompatibilityAxis, ...],
    dict[tuple[tuple[str, ...], str], int],
]:
    offsets: dict[tuple[tuple[str, ...], str], int] = {}
    ambient_basis: list[SheafCochainCoordinate] = []
    for stalk in plan.sheaf.stalks:
        for label in stalk.basis:
            offsets[(stalk.simplex, label)] = len(ambient_basis)
            ambient_basis.append(
                SheafCochainCoordinate(simplex=stalk.simplex, basis_label=label)
            )

    zero = plan.field.zero()
    one = plan.field.one()
    scalar_rows: list[list[object]] = []
    row_axes: list[SheafSectionCompatibilityAxis] = []
    for source, target in plan.pairs:
        restriction_matrix = parsed[(source, target)]
        source_basis = plan.stalk_for[source].basis
        target_basis = plan.stalk_for[target].basis
        for row, target_label in enumerate(target_basis):
            coefficients = [zero for _ in range(plan.ambient_dimension)]
            for column, source_label in enumerate(source_basis):
                coefficients[offsets[(source, source_label)]] = restriction_matrix[row][
                    column
                ]
            coefficients[offsets[(target, target_label)]] = -one
            scalar_rows.append(coefficients)
            row_axes.append(
                SheafSectionCompatibilityAxis(
                    source=source, target=target, target_basis_label=target_label
                )
            )

    return tuple(ambient_basis), scalar_rows, tuple(row_axes), offsets


def _section_evaluations(
    plan: _SectionPlan,
    offsets: dict[tuple[tuple[str, ...], str], int],
    section_vectors: list[list[object]],
    section_basis: tuple[str, ...],
) -> tuple[SheafSectionEvaluation, ...]:
    evaluations: list[SheafSectionEvaluation] = []
    for stalk in plan.sheaf.stalks:
        entries = tuple(
            tuple(
                plan.field.typed(
                    section_vectors[column][offsets[(stalk.simplex, label)]]
                )
                for column in range(len(section_basis))
            )
            for label in stalk.basis
        )
        evaluations.append(
            SheafSectionEvaluation(
                simplex=stalk.simplex,
                stalk_basis=stalk.basis,
                section_basis=section_basis,
                entries=entries,
            )
        )
    return tuple(evaluations)


def sections(sheaf: FiniteCellularSheaf) -> SheafSectionSpace:
    """Return the exact kernel of all stalk-compatibility equations."""
    plan = _admit_section_plan(sheaf)
    parsed, input_chars, max_input_chars, max_input_digits = (
        _parse_section_restrictions(plan)
    )
    result_scalar_digits = _require_section_height_bound(
        plan, input_chars, max_input_chars, max_input_digits
    )
    ambient_basis, scalar_rows, row_axes, offsets = _section_matrix_and_axes(
        plan, parsed
    )
    section_vectors = _cochain_nullspace(
        plan.field, scalar_rows, plan.ambient_dimension
    )
    dimension = len(section_vectors)
    output_cells = (
        plan.matrix_cells + 2 * plan.ambient_dimension * dimension + plan.row_count
    )
    if output_cells > MAX_SHEAF_SECTION_OUTPUT_CELLS:
        raise _section_resource(
            "output_cells_bound",
            "the exact section-space output exceeds its cell bound",
        )
    matrix_text = tuple(
        tuple(plan.field.typed(value) for value in row) for row in scalar_rows
    )
    basis_coordinates = tuple(
        tuple(plan.field.typed(value) for value in vector) for vector in section_vectors
    )
    section_basis = tuple(f"section_{index}" for index in range(dimension))
    evaluations = _section_evaluations(plan, offsets, section_vectors, section_basis)
    scalar_count = sum(
        len(row) for matrix in (matrix_text, basis_coordinates) for row in matrix
    )
    scalar_count += sum(
        len(row) for evaluation in evaluations for row in evaluation.entries
    )
    scalar_digits = (
        max_input_digits
        if plan.sheaf.coefficient_field.value != "QQ"
        else result_scalar_digits
    )
    serialized_chars = sheaf_scalar_json_bound(scalar_count, scalar_digits)
    if serialized_chars > MAX_SHEAF_SECTION_OUTPUT_CHARS:
        raise _section_resource(
            "output_chars_bound",
            f"section-space exact scalar output exceeds {MAX_SHEAF_SECTION_OUTPUT_CHARS} characters",
        )

    return SheafSectionSpace._from_kernel(
        sheaf=plan.sheaf,
        dimension=dimension,
        section_basis=section_basis,
        ambient_basis=ambient_basis,
        compatibility_row_axes=row_axes,
        compatibility_matrix=matrix_text,
        basis_coordinates=basis_coordinates,
        evaluations=evaluations,
    )


def restrict_sections(
    sheaf: FiniteCellularSheaf, subcomplex: FiniteSimplicialComplex
) -> SheafSectionRestriction:
    """Compute restriction of global sections to an included subcomplex."""
    restricted = restrict_to_subcomplex(sheaf, subcomplex).subcomplex
    # Each section-space computation owns and performs its ordinary admission.
    source = sections(sheaf)
    target = sections(restricted)
    source_offsets: dict[tuple[str, ...], int] = {}
    offset = 0
    for stalk in source.sheaf.stalks:
        source_offsets[stalk.simplex] = offset
        offset += len(stalk.basis)
    target_positions = tuple(
        source_offsets[cell] + coordinate
        for stalk in target.sheaf.stalks
        for cell in (stalk.simplex,)
        for coordinate in range(len(stalk.basis))
    )
    field = _admit_field(source.sheaf.coefficient_field, source.sheaf.prime)
    source_vectors = [
        [field.parse(value) for value in vector] for vector in source.basis_coordinates
    ]
    target_vectors = [
        [field.parse(value) for value in vector] for vector in target.basis_coordinates
    ]
    restricted_columns = [
        [vector[index] for index in target_positions] for vector in source_vectors
    ]
    # Solve T^T c = r in canonical target section coordinates. Inclusion of
    # subcomplexes guarantees r lies in the target section space.
    target_rows = [
        [target_vectors[column][row] for column in range(len(target_vectors))]
        for row in range(len(target_positions))
    ]
    output_cells = source.dimension * target.dimension
    if output_cells > MAX_SHEAF_SECTION_OUTPUT_CELLS:
        raise _section_resource(
            "restriction_matrix_bound",
            "section restriction matrix exceeds its cell bound",
        )
    work_bound = len(
        target_positions
    ) * target.dimension * target.dimension + output_cells * max(
        1, len(target_positions)
    )
    if work_bound > MAX_SHEAF_SECTION_WORK:
        raise _section_resource(
            "restriction_work_bound",
            "exact section restriction coordinate work exceeds its bound",
        )
    max_basis_scalar_chars = max(
        (
            sheaf_scalar_digits(value)
            for space in (source, target)
            for vector in space.basis_coordinates
            for value in vector
        ),
        default=1,
    )
    if source.sheaf.coefficient_field.value == "QQ":
        map_scalar_chars = (
            2
            * (
                target.dimension * max_basis_scalar_chars
                + len(str(factorial(target.dimension)))
            )
            + 3
        )
    else:
        map_scalar_chars = len(str(source.sheaf.prime))
    output_bound = (
        len(source.model_dump_json())
        + len(target.model_dump_json())
        + output_cells * map_scalar_chars
    )
    if output_bound > MAX_SHEAF_SECTION_RESTRICTION_OUTPUT_CHARS:
        raise _section_resource(
            "restriction_output_bound",
            "source, target, and induced section map exceed the aggregate result bound",
        )
    columns: list[list[object]] = []
    for restricted in restricted_columns:
        augmented = [
            [*row, value] for row, value in zip(target_rows, restricted, strict=True)
        ]
        reduced, pivots = _cochain_rref(field, augmented)
        width = target.dimension + 1
        if any(
            all(value == 0 for value in row[: target.dimension])
            and row[target.dimension] != 0
            for row in reduced
        ):
            raise RuntimeError("subcomplex section restriction left the target kernel")
        coordinates = [field.zero() for _ in range(target.dimension)]
        for row_index, pivot in enumerate(pivots):
            if pivot < target.dimension:
                coordinates[pivot] = reduced[row_index][width - 1]
        columns.append(coordinates)
    matrix = tuple(
        tuple(field.typed(columns[column][row]) for column in range(len(columns)))
        for row in range(target.dimension)
    )
    return SheafSectionRestriction._from_kernel(
        source=source, target=target, entries=matrix
    )


def restriction(
    sheaf: FiniteCellularSheaf, source: tuple[str, ...], target: tuple[str, ...]
) -> SheafRestrictionResult:
    source = tuple(sorted(source))
    target = tuple(sorted(target))
    for item in (*sheaf.cover_restrictions, *sheaf.derived_restrictions):
        if item.source == source and item.target == target:
            return SheafRestrictionResult(sheaf=sheaf, restriction=item)
    raise OperationDomainValidationError(
        location=("source", "target"),
        code="cellular_sheaf.restriction_missing",
        message="the requested comparable restriction is not present",
    )


def _complete_diagram(sheaf: FiniteCellularSheaf, *, role: str) -> None:
    faces = sheaf.canonical_face_order
    expected = {
        (source, target)
        for source in faces
        for target in faces
        if len(target) > len(source) and set(source).issubset(target)
    }
    actual = {
        (restriction.source, restriction.target)
        for restriction in (*sheaf.cover_restrictions, *sheaf.derived_restrictions)
    }
    if actual != expected:
        raise OperationDomainValidationError(
            location=(role,),
            code="cellular_sheaf.morphism_incomplete_diagram",
            message="both sheaves must carry every canonical comparable restriction",
        )


def _scan_morphism_scalar_text(
    source: FiniteCellularSheaf,
    target: FiniteCellularSheaf,
    components: tuple[Component, ...],
) -> tuple[int, int]:
    field = _admit_field(source.coefficient_field, source.prime)
    total_chars = 0
    max_digits = 1
    for _key, matrix in components:
        for row in matrix:
            for value in row:
                if not isinstance(value, CanonicalRational) and type(value) is not int:
                    raise _section_domain(
                        "morphism.scalar_invalid",
                        "components must contain exact scalars in the declared typed field",
                    )
                digits = sheaf_scalar_digits(value)
                try:
                    field.parse(value)
                except ValueError as exc:
                    raise _section_domain(
                        "morphism.scalar_invalid",
                        "components must use the source sheaf's canonical scalar type",
                    ) from exc
                if digits > MAX_SHEAF_ENTRY_DIGITS:
                    raise _section_resource(
                        "morphism.coefficient_digits_bound",
                        f"a component coefficient exceeds {MAX_SHEAF_ENTRY_DIGITS} digits",
                    )
                total_chars += sheaf_scalar_json_bound(1, digits)
                max_digits = max(max_digits, digits)
    for parent in (source, target):
        for restriction in (*parent.cover_restrictions, *parent.derived_restrictions):
            for row in restriction.entries:
                for value in row:
                    if (
                        not isinstance(value, CanonicalRational)
                        and type(value) is not int
                    ):
                        raise _section_domain(
                            "morphism.scalar_invalid",
                            "restriction maps must contain exact scalars in the declared typed field",
                        )
                    digits = sheaf_scalar_digits(value)
                    try:
                        field.parse(value)
                    except ValueError as exc:
                        raise _section_domain(
                            "morphism.scalar_invalid",
                            "restriction maps must use the source sheaf's canonical scalar type",
                        ) from exc
                    if digits > MAX_SHEAF_ENTRY_DIGITS:
                        raise _section_resource(
                            "morphism.coefficient_digits_bound",
                            f"a restriction coefficient exceeds {MAX_SHEAF_ENTRY_DIGITS} digits",
                        )
                    total_chars += sheaf_scalar_json_bound(1, digits)
                    max_digits = max(max_digits, digits)
    return total_chars, max_digits


def _admit_morphism_resources(
    source: FiniteCellularSheaf,
    target: FiniteCellularSheaf,
    components: tuple[Component, ...],
) -> dict:
    axis = source.canonical_face_order
    source_stalks = {stalk.simplex: stalk for stalk in source.stalks}
    target_stalks = {stalk.simplex: stalk for stalk in target.stalks}
    if len(source_stalks) != len(axis) or len(target_stalks) != len(axis):
        raise _section_domain(
            "morphism.stalk_axis", "each parent must bind one stalk to every simplex"
        )
    component_cells = sum(
        len(source_stalks[cell].basis) * len(target_stalks[cell].basis) for cell in axis
    )
    if component_cells > MAX_SHEAF_MORPHISM_COMPONENT_CELLS:
        raise _section_resource(
            "morphism.component_cells_bound",
            "pointwise component matrices exceed the admitted cell count",
        )
    target_cover = {
        (item.source, item.target): item for item in target.cover_restrictions
    }
    square_work = 0
    for restriction in source.cover_restrictions:
        a, b = restriction.source, restriction.target
        if (a, b) not in target_cover:
            raise _section_domain(
                "morphism.cover_axis", "both sheaves must bind the same cover maps"
            )
        f_a, f_b = len(source_stalks[a].basis), len(source_stalks[b].basis)
        g_a, g_b = len(target_stalks[a].basis), len(target_stalks[b].basis)
        square_work += g_b * f_a * (g_a + f_b)
    total_input_chars, max_input_digits = _scan_morphism_scalar_text(
        source, target, components
    )
    if square_work > MAX_SHEAF_MORPHISM_WORK:
        raise _section_resource(
            "morphism.work_bound", "naturality-square arithmetic exceeds its work bound"
        )
    scalar_growth = (2 * MAX_SHEAF_STALK_RANK * max_input_digits + 8) * (
        MAX_SHEAF_STALK_RANK + 1
    )
    if (
        total_input_chars + square_work * scalar_growth
        > MAX_SHEAF_MORPHISM_OUTPUT_CHARS
    ):
        raise _section_resource(
            "morphism.output_chars_bound",
            "worst-case exact naturality arithmetic exceeds its output envelope",
        )
    return target_cover


def _admit_component_matrix(
    matrix: Any, field: Any, location: tuple[str | int, ...]
) -> tuple[tuple[Any, ...], ...]:
    """Parse every caller scalar before inspecting component axes."""
    try:
        parsed = tuple(tuple(field.parse(entry) for entry in row) for row in matrix)
    except (
        AttributeError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
    ) as exc:
        raise OperationDomainValidationError(
            location=location,
            code="cellular_sheaf.morphism_scalar_invalid",
            message="sheaf components must contain valid exact scalars",
        ) from exc
    return parsed


def _mul(a: Any, b: Any, p: int | None, *, output_width: int) -> Any:
    if not a:
        return []
    if not b:
        return [[0] * output_width for _ in a]
    cols = list(zip(*b, strict=False))
    result = []
    for row in a:
        result_row = []
        for col in cols:
            value = sum(x * y for x, y in zip(row, col, strict=False))
            result_row.append(value % p if p is not None else value)
        result.append(result_row)
    return result


def _resolve_component_key(
    key: ComponentKey, simplices: tuple[tuple[str, ...], ...]
) -> tuple[str, ...]:
    if isinstance(key, tuple):
        return key
    matches = tuple(simplex for simplex in simplices if ".".join(simplex) == key)
    if len(matches) != 1:
        raise OperationDomainValidationError(
            location=("components",),
            code="cellular_sheaf.morphism_component_axis",
            message=(
                "dotted component keys must identify exactly one stalk; use "
                "canonical simplex tuple keys when labels contain dots"
            ),
        )
    return matches[0]


def _transpose(a: Any) -> Any:
    return list(map(list, zip(*a, strict=False))) if a else []


def morphism(
    source: FiniteCellularSheaf,
    target: FiniteCellularSheaf,
    components: tuple[Component, ...],
) -> SheafMorphismResult:
    # FiniteCellularSheaf parsing intentionally remains structural: a parsed
    # carrier (and especially one made with model_construct) does not prove
    # that its declared GF(p) modulus is prime.  Establish both fields before
    # inspecting carrier internals or performing arithmetic so malformed
    # caller-authored carriers cannot leak raw attribute/type errors.
    source_field = _admit_field(source.coefficient_field, source.prime)
    _admit_field(target.coefficient_field, target.prime)
    if (
        source.complex != target.complex
        or source.coefficient_field != target.coefficient_field
        or source.prime != target.prime
    ):
        raise OperationDomainValidationError(
            location=("target",),
            code="cellular_sheaf.morphism.parent_mismatch",
            message="sheaf morphisms require one complex and coefficient field",
        )
    _complete_diagram(source, role="source")
    _complete_diagram(target, role="target")
    target_cover = _admit_morphism_resources(source, target, components)
    p = source_field.prime
    source_axis = source.canonical_face_order
    normalized_keys = tuple(
        _resolve_component_key(key, source_axis) for key, _matrix in components
    )
    expected = source_axis
    if normalized_keys != expected:
        raise OperationDomainValidationError(
            location=("components",),
            code="cellular_sheaf.morphism_component_axis",
            message="one component in canonical stalk order is required",
        )
    given = {}
    canonical_components = []
    for key, (_raw_key, matrix) in zip(normalized_keys, components, strict=True):
        parsed_matrix = _admit_component_matrix(
            matrix, source_field, ("components", ".".join(key))
        )
        given[key] = parsed_matrix
        canonical_components.append((key, source_field.render(parsed_matrix)))
    stalk = {s.simplex: s for s in source.stalks}
    target_stalk = {s.simplex: s for s in target.stalks}
    for key in expected:
        matrix = given[key]
        rows = len(target_stalk[key].basis)
        cols = len(stalk[key].basis)
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            raise OperationDomainValidationError(
                location=("components", ".".join(key)),
                code="cellular_sheaf.morphism_shape",
                message="component axes do not match stalk ranks",
            )
    try:
        for restriction in source.cover_restrictions:
            a = restriction.source
            b = restriction.target
            left = _mul(
                [list(row) for row in given[b]],
                [[source_field.parse(x) for x in row] for row in restriction.entries],
                p,
                output_width=len(stalk[a].basis),
            )
            tr = target_cover[(restriction.source, restriction.target)]
            right = _mul(
                [[source_field.parse(x) for x in row] for row in tr.entries],
                [list(row) for row in given[a]],
                p,
                output_width=len(stalk[a].basis),
            )
            if left != right:
                return SheafMorphismResult(
                    source=source,
                    target=target,
                    components=tuple(canonical_components),
                    natural=False,
                    obstruction=f"naturality fails on {a} < {b}",
                )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise OperationDomainValidationError(
            location=("components",),
            code="cellular_sheaf.morphism_scalar_invalid",
            message="sheaf components and restrictions must contain valid exact scalars",
        ) from error
    return SheafMorphismResult(
        source=source,
        target=target,
        components=tuple(canonical_components),
        natural=True,
    )


def compose_morphisms(
    first: SheafMorphismResult, second: SheafMorphismResult
) -> SheafMorphismResult:
    """Compose two source-bound natural maps after admitting their claims.

    Serialized maps are caller-supplied values, so each input's naturality is
    re-established here before it is used. The composite then follows from
    exact pointwise matrix multiplication and needs no third square replay.
    """
    checked_first = morphism(first.source, first.target, first.components)
    checked_second = morphism(second.source, second.target, second.components)
    if not checked_first.natural or not checked_second.natural:
        raise _section_domain(
            "morphism.compose_non_natural",
            "composition requires two natural cellular-sheaf morphisms",
        )
    if first.target != second.source:
        raise _section_domain(
            "morphism.compose_parent_mismatch",
            "the target sheaf of the first map must equal the source of the second",
        )
    field = _admit_field(first.source.coefficient_field, first.source.prime)
    by_first = dict(checked_first.components)
    by_second = dict(checked_second.components)
    source_stalks = {item.simplex: len(item.basis) for item in first.source.stalks}
    middle_stalks = {item.simplex: len(item.basis) for item in first.target.stalks}
    final_stalks = {item.simplex: len(item.basis) for item in second.target.stalks}
    total_work = sum(
        source_stalks[cell] * middle_stalks[cell] * final_stalks[cell]
        for cell in first.source.canonical_face_order
    )
    output_cells = sum(
        source_stalks[cell] * final_stalks[cell]
        for cell in first.source.canonical_face_order
    )
    max_input_digits = max(
        (
            sheaf_scalar_digits(value)
            for matrix in (*by_first.values(), *by_second.values())
            for row in matrix
            for value in row
        ),
        default=1,
    )
    scalar_chars_bound = sheaf_scalar_json_bound(
        output_cells,
        (2 * MAX_SHEAF_STALK_RANK * max_input_digits + 8) * MAX_SHEAF_STALK_RANK,
    )
    if (
        total_work > MAX_SHEAF_MORPHISM_WORK
        or scalar_chars_bound > MAX_SHEAF_MORPHISM_OUTPUT_CHARS
    ):
        raise _section_resource(
            "morphism.compose_bound",
            "composite component arithmetic exceeds its admitted work or output bound",
        )
    composed: list[Component] = []
    output_chars = 0
    for simplex in first.source.canonical_face_order:
        left = tuple(tuple(field.parse(x) for x in row) for row in by_first[simplex])
        right = tuple(tuple(field.parse(x) for x in row) for row in by_second[simplex])
        source_rank = source_stalks[simplex]
        final_rank = final_stalks[simplex]
        if final_rank == 0:
            matrix = ()
        elif source_rank == 0:
            matrix = tuple(() for _ in range(final_rank))
        elif middle_stalks[simplex] == 0:
            matrix = tuple(
                tuple(field.zero() for _ in range(source_rank))
                for _ in range(final_rank)
            )
        else:
            matrix = field.matmul(right, left)
        rendered = field.render(matrix)
        output_chars += sheaf_scalar_json_bound(
            sum(len(row) for row in rendered),
            max(
                (sheaf_scalar_digits(value) for row in rendered for value in row),
                default=1,
            ),
        )
        composed.append((simplex, rendered))
    if output_chars > MAX_SHEAF_MORPHISM_OUTPUT_CHARS:
        raise _section_resource(
            "morphism.compose_bound",
            "composite component arithmetic exceeds its admitted bound",
        )
    return SheafMorphismResult(
        source=first.source,
        target=second.target,
        components=tuple(composed),
        natural=True,
    )


__all__ = [
    "SheafMorphismComposeRequest",
    "SheafMorphismRequest",
    "SheafMorphismResult",
    "SheafRestrictionRequest",
    "SheafRestrictionResult",
    "SheafSectionCompatibilityAxis",
    "SheafSectionEvaluation",
    "SheafSectionRestriction",
    "SheafSectionRestrictionRequest",
    "SheafSectionSpace",
    "SheafSectionsRequest",
    "compose_morphisms",
    "morphism",
    "restrict_sections",
    "restriction",
    "sections",
]
