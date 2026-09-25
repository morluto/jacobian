"""Exact pointwise cokernels of finite cellular sheaf morphisms."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cellular_sheaves._kernel import (
    Scalar,
    _admit_field,
    _cochain_rref,
    _ExactField,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_MORPHISM_OUTPUT_CHARS,
    MAX_SHEAF_MORPHISM_WORK,
    FiniteCellularSheaf,
    SheafRestriction,
    SheafScalar,
    SheafStalk,
    _require_field_scalars,
    sheaf_scalar_json_bound,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    Component,
    SheafMorphismResult,
    _admit_component_matrix,
    _admit_morphism_resources,
    _admit_section_plan,
    _mul,
    _resolve_component_key,
)
from jacobian.math.topology.cellular_sheaves.morphism_kernel import (
    _parent_reconstruction_bounds,
    _readmit_parent_sheaf,
)


class SheafMorphismCokernelRequest(StrictModel):
    """A natural sheaf morphism whose pointwise cokernel is requested."""

    morphism: SheafMorphismResult


class SheafCokernelStalkLift(StrictModel):
    """Chosen target-stalk representatives of the quotient basis classes."""

    simplex: tuple[str, ...]
    entries: tuple[tuple[SheafScalar, ...], ...]


class SheafMorphismCokernelResult(StrictModel):
    """Cokernel sheaf and its canonical quotient map from the target."""

    morphism: SheafMorphismResult
    cokernel: FiniteCellularSheaf
    projection: SheafMorphismResult
    stalk_lifts: tuple[SheafCokernelStalkLift, ...]

    @model_validator(mode="after")
    def bind_cokernel_projection(self) -> Self:
        source, target = self.morphism.source, self.morphism.target
        if (
            self.cokernel.complex != source.complex
            or self.cokernel.coefficient_field != source.coefficient_field
            or self.cokernel.prime != source.prime
        ):
            raise ValueError("cokernel must retain the morphism complex and field")
        if self.projection.source != target or self.projection.target != self.cokernel:
            raise ValueError("projection must map the target sheaf onto the cokernel")
        cells = source.canonical_face_order
        if tuple(stalk.simplex for stalk in self.cokernel.stalks) != cells:
            raise ValueError("cokernel stalks must retain canonical simplex axes")
        if not self.projection.natural or self.projection.obstruction is not None:
            raise ValueError("cokernel projection must be natural")
        if tuple(key for key, _matrix in self.projection.components) != cells:
            raise ValueError("projection components must retain canonical simplex axes")
        target_ranks = {stalk.simplex: len(stalk.basis) for stalk in target.stalks}
        quotient_ranks = {
            stalk.simplex: len(stalk.basis) for stalk in self.cokernel.stalks
        }
        for cell, matrix in self.projection.components:
            if not isinstance(cell, tuple):
                raise ValueError("projection components require simplex tuple axes")
            if len(matrix) != quotient_ranks[cell] or any(
                len(row) != target_ranks[cell] for row in matrix
            ):
                raise ValueError("projection matrices must match target and quotient axes")
        if tuple(lift.simplex for lift in self.stalk_lifts) != cells:
            raise ValueError("quotient lifts must retain canonical simplex axes")
        for lift in self.stalk_lifts:
            if len(lift.entries) != target_ranks[lift.simplex] or any(
                len(row) != quotient_ranks[lift.simplex] for row in lift.entries
            ):
                raise ValueError("quotient lifts must match target and quotient axes")
        _require_field_scalars(
            tuple(lift.entries for lift in self.stalk_lifts),
            source.coefficient_field,
            source.prime,
            label="cokernel stalk lift",
        )
        return self

    @classmethod
    def _from_cokernel(
        cls,
        *,
        morphism: SheafMorphismResult,
        cokernel: FiniteCellularSheaf,
        projection: SheafMorphismResult,
        stalk_lifts: tuple[SheafCokernelStalkLift, ...],
    ) -> Self:
        """Build the admitted result without replaying quotient calculations."""
        return cls.model_construct(
            morphism=morphism,
            cokernel=cokernel,
            projection=projection,
            stalk_lifts=stalk_lifts,
        )


def _domain(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("morphism",),
        code=f"topology.cellular_sheaf.morphism_cokernel.{code}",
        message=message,
    )


def _resource(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("morphism",),
        code=f"topology.cellular_sheaf.morphism_cokernel.{code}",
        message=message,
    )


def _quotient_coordinates(
    field: _ExactField,
    component: tuple[tuple[Scalar, ...], ...],
    target_rank: int,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[Scalar, ...], ...],
    tuple[tuple[Scalar, ...], ...],
]:
    """Return pivot/free target axes, quotient projection, and complement lift."""
    source_rank = len(component[0]) if component else 0
    columns = tuple(
        tuple(component[row][column] for column in range(source_rank))
        for row in range(target_rank)
    )
    transposed = [
        [columns[row][column] for row in range(target_rank)]
        for column in range(source_rank)
    ]
    reduced, pivots = _cochain_rref(field, transposed)
    pivot_rows = tuple(pivots)
    free_rows = tuple(row for row in range(target_rank) if row not in pivot_rows)
    row_by_pivot = {pivot: row for row, pivot in enumerate(pivot_rows)}

    projection = []
    for free in free_rows:
        row = [field.zero() for _ in range(target_rank)]
        row[free] = field.one()
        for pivot in pivot_rows:
            row[pivot] = -reduced[row_by_pivot[pivot]][free]
            if field.prime is not None:
                row[pivot] %= field.prime
        projection.append(tuple(row))

    lift = []
    for coordinate in range(target_rank):
        lift.append(
            tuple(
                field.one() if coordinate == free else field.zero()
                for free in free_rows
            )
        )
    return pivot_rows, tuple(projection), tuple(lift)


def cokernel_of_morphism(
    value: SheafMorphismResult,
) -> SheafMorphismCokernelResult:
    """Compute the pointwise quotient sheaf and its target projection."""
    source, target = value.source, value.target
    field = _admit_field(source.coefficient_field, source.prime)
    _admit_field(target.coefficient_field, target.prime)
    if (
        source.complex != target.complex
        or source.coefficient_field != target.coefficient_field
        or source.prime != target.prime
    ):
        raise _domain("parent_mismatch", "morphisms require one complex and field")
    _admit_section_plan(source)
    _admit_section_plan(target)
    if not isinstance(value.components, tuple) or any(
        not isinstance(component, tuple)
        or len(component) != 2
        or not isinstance(component[1], (tuple, list))
        for component in value.components
    ):
        raise _domain("component_structure", "components must be simplex-matrix pairs")
    target_cover, input_digits, morphism_work = _admit_morphism_resources(
        source, target, value.components
    )
    cells = source.canonical_face_order
    keys = tuple(_resolve_component_key(key, cells) for key, _matrix in value.components)
    if keys != cells:
        raise _domain("component_axis", "one component per simplex in canonical order is required")
    source_stalks = {stalk.simplex: stalk for stalk in source.stalks}
    target_stalks = {stalk.simplex: stalk for stalk in target.stalks}
    components: dict[tuple[str, ...], tuple[tuple[Scalar, ...], ...]] = {}
    canonical_components: list[Component] = []
    for cell, (_key, raw) in zip(cells, value.components, strict=True):
        matrix = _admit_component_matrix(raw, field, ("components", ".".join(cell)))
        if len(matrix) != len(target_stalks[cell].basis) or any(
            len(row) != len(source_stalks[cell].basis) for row in matrix
        ):
            raise _domain("component_shape", "components must match the stalk axes")
        components[cell] = matrix
        canonical_components.append((cell, field.render(matrix)))

    target_restrictions = {
        (item.source, item.target): item
        for item in (*target.cover_restrictions, *target.derived_restrictions)
    }
    restriction_cells = sum(
        len(target_stalks[lower].basis) * len(target_stalks[upper].basis)
        for lower, upper in target_restrictions
    )
    work = morphism_work
    max_rank = max((len(stalk.basis) for stalk in target.stalks), default=0)
    work += sum(
        max(1, 2 * len(target_stalks[cell].basis) ** 2 * len(source_stalks[cell].basis))
        for cell in cells
    )
    work += sum(
        max(
            1,
            4
            * len(target_stalks[lower].basis)
            * len(target_stalks[upper].basis)
            * max_rank,
        )
        for lower, upper in target_restrictions
    )
    reconstruction_work, reconstruction_chars = _parent_reconstruction_bounds(
        (source, target), input_digits=input_digits
    )
    if work + reconstruction_work > MAX_SHEAF_MORPHISM_WORK:
        raise _resource("work_bound", "cokernel and parent reconstruction exceed work bounds")
    output_cells = (
        restriction_cells
        + sum(len(stalk.basis) ** 2 for stalk in target.stalks)
        + sum(len(stalk.basis) ** 2 for stalk in target.stalks)
    )
    rank_growth = max(1, max_rank)
    output_digits = (
        2 * rank_growth**2 * input_digits + 2 * rank_growth**2 + 32
        if source.coefficient_field.value == "QQ"
        else max(input_digits, len(str(source.prime)))
    )
    if (
        4 * len(value.model_dump_json())
        + reconstruction_chars
        + sheaf_scalar_json_bound(output_cells, output_digits)
        > MAX_SHEAF_MORPHISM_OUTPUT_CHARS
    ):
        raise _resource("output_bound", "cokernel quotient maps exceed the output envelope")

    _readmit_parent_sheaf(source, role="source")
    _readmit_parent_sheaf(target, role="target")
    for item in source.cover_restrictions:
        source_map = tuple(tuple(field.parse(x) for x in row) for row in item.entries)
        target_item = target_cover[(item.source, item.target)]
        target_map = tuple(tuple(field.parse(x) for x in row) for row in target_item.entries)
        left = _mul(
            [list(row) for row in components[item.target]],
            [list(row) for row in source_map],
            field.prime,
            output_width=len(source_stalks[item.source].basis),
        )
        right = _mul(
            [list(row) for row in target_map],
            [list(row) for row in components[item.source]],
            field.prime,
            output_width=len(source_stalks[item.source].basis),
        )
        if left != right:
            raise _domain("morphism_not_natural", "cokernel requires a natural morphism")
    checked = SheafMorphismResult(
        source=source,
        target=target,
        components=tuple(canonical_components),
        natural=True,
    )

    quotient_ranks: dict[tuple[str, ...], int] = {}
    quotient_projections: dict[tuple[str, ...], tuple[tuple[Scalar, ...], ...]] = {}
    quotient_lifts: dict[tuple[str, ...], tuple[tuple[Scalar, ...], ...]] = {}
    quotient_stalks = []
    projection_components: list[Component] = []
    stalk_lifts: list[SheafCokernelStalkLift] = []
    for cell in cells:
        target_rank = len(target_stalks[cell].basis)
        _pivots, quotient_projection, lift = _quotient_coordinates(
            field, components[cell], target_rank
        )
        quotient_projections[cell] = quotient_projection
        quotient_lifts[cell] = lift
        rank = len(quotient_projection)
        quotient_ranks[cell] = rank
        quotient_stalks.append(
            SheafStalk(simplex=cell, basis=tuple(f"c{i}" for i in range(rank)))
        )
        stalk_lifts.append(
            SheafCokernelStalkLift(simplex=cell, entries=field.render(lift))
        )
        annihilated = _mul(
            [list(row) for row in quotient_projection],
            [list(row) for row in components[cell]],
            field.prime,
            output_width=len(source_stalks[cell].basis),
        )
        if any(value != 0 for row in annihilated for value in row):
            raise ArithmeticError("cokernel projection does not annihilate the morphism")
        projection_components.append((cell, field.render(quotient_projection)))

    def induced(item: SheafRestriction) -> SheafRestriction:
        restriction = tuple(tuple(field.parse(x) for x in row) for row in item.entries)
        left = _mul(
            [list(row) for row in quotient_projections[item.target]],
            [list(row) for row in restriction],
            field.prime,
            output_width=len(target_stalks[item.source].basis),
        )
        entries = _mul(
            left,
            [list(row) for row in quotient_lifts[item.source]],
            field.prime,
            output_width=quotient_ranks[item.source],
        )
        reconstructed = _mul(
            entries,
            [list(row) for row in quotient_projections[item.source]],
            field.prime,
            output_width=len(target_stalks[item.source].basis),
        )
        if reconstructed != left:
            raise ArithmeticError("cokernel projection is not natural")
        return SheafRestriction(
            source=item.source,
            target=item.target,
            row_basis=tuple(f"c{i}" for i in range(quotient_ranks[item.target])),
            column_basis=tuple(f"c{i}" for i in range(quotient_ranks[item.source])),
            entries=field.render(tuple(tuple(row) for row in entries)),
            cover_path=item.cover_path,
        )

    quotient = FiniteCellularSheaf._from_kernel(
        complex=source.complex,
        coefficient_field=source.coefficient_field,
        prime=source.prime,
        stalks=tuple(quotient_stalks),
        cover_restrictions=tuple(induced(item) for item in target.cover_restrictions),
        derived_restrictions=tuple(induced(item) for item in target.derived_restrictions),
        diamonds=target.diamonds,
        comparable_pairs=target.comparable_pairs,
    )
    projection_map = SheafMorphismResult(
        source=target,
        target=quotient,
        components=tuple(projection_components),
        natural=True,
    )
    return SheafMorphismCokernelResult._from_cokernel(
        morphism=checked,
        cokernel=quotient,
        projection=projection_map,
        stalk_lifts=tuple(stalk_lifts),
    )


__all__ = [
    "SheafCokernelStalkLift",
    "SheafMorphismCokernelRequest",
    "SheafMorphismCokernelResult",
    "cokernel_of_morphism",
]
