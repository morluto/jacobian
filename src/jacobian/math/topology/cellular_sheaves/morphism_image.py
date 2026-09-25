"""Exact pointwise images of finite cellular sheaf morphisms."""

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
    MAX_SHEAF_STALK_RANK,
    FiniteCellularSheaf,
    SheafField,
    SheafRestriction,
    SheafStalk,
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


class SheafMorphismImageRequest(StrictModel):
    """A natural sheaf morphism whose pointwise image is requested."""

    morphism: SheafMorphismResult


class SheafMorphismImageResult(StrictModel):
    """Image sheaf, its target inclusion, and the canonical factorization."""

    morphism: SheafMorphismResult
    image: FiniteCellularSheaf
    inclusion: SheafMorphismResult
    factor: SheafMorphismResult

    @model_validator(mode="after")
    def bind_image_factorization(self) -> Self:
        source, target = self.morphism.source, self.morphism.target
        if (
            self.image.complex != source.complex
            or self.image.coefficient_field != source.coefficient_field
            or self.image.prime != source.prime
        ):
            raise ValueError("image must retain the morphism's complex and field")
        if self.inclusion.source != self.image or self.inclusion.target != target:
            raise ValueError("inclusion must map the image sheaf into the target")
        if self.factor.source != source or self.factor.target != self.image:
            raise ValueError("factor must map the source onto the image sheaf")
        cells = source.canonical_face_order
        if tuple(stalk.simplex for stalk in self.image.stalks) != cells:
            raise ValueError("image stalks must retain the canonical simplex axes")
        source_ranks = {stalk.simplex: len(stalk.basis) for stalk in source.stalks}
        target_ranks = {stalk.simplex: len(stalk.basis) for stalk in target.stalks}
        image_ranks = {stalk.simplex: len(stalk.basis) for stalk in self.image.stalks}
        for map_ in (self.inclusion, self.factor):
            if not map_.natural or map_.obstruction is not None:
                raise ValueError("image factorization maps must be natural")
            if tuple(key for key, _matrix in map_.components) != cells:
                raise ValueError(
                    "image factorization components must retain simplex axes"
                )
        for cell, matrix in self.inclusion.components:
            if not isinstance(cell, tuple):
                raise ValueError("image inclusion components need simplex tuple axes")
            if len(matrix) != target_ranks[cell] or any(
                len(row) != image_ranks[cell] for row in matrix
            ):
                raise ValueError("image inclusion matrices must match the stalk axes")
        for cell, matrix in self.factor.components:
            if not isinstance(cell, tuple):
                raise ValueError("image factor components need simplex tuple axes")
            if len(matrix) != image_ranks[cell] or any(
                len(row) != source_ranks[cell] for row in matrix
            ):
                raise ValueError("image factor matrices must match the stalk axes")
        return self

    @classmethod
    def _from_image(
        cls,
        *,
        morphism: SheafMorphismResult,
        image: FiniteCellularSheaf,
        inclusion: SheafMorphismResult,
        factor: SheafMorphismResult,
    ) -> Self:
        """Build an admitted result without replaying the computed factorization."""
        return cls.model_construct(
            morphism=morphism, image=image, inclusion=inclusion, factor=factor
        )


def _domain(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("morphism",),
        code=f"topology.cellular_sheaf.morphism_image.{code}",
        message=message,
    )


def _resource(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("morphism",),
        code=f"topology.cellular_sheaf.morphism_image.{code}",
        message=message,
    )


def _independent_columns(
    field: _ExactField, matrix: tuple[tuple[Scalar, ...], ...], width: int
) -> tuple[int, ...]:
    """Return canonical pivot columns, including for empty rectangular maps."""
    if not matrix:
        return ()
    _reduced, pivots = _cochain_rref(field, [list(row) for row in matrix])
    return tuple(pivot for pivot in pivots if pivot < width)


def _solve_matrix(
    field: _ExactField,
    basis: tuple[tuple[Scalar, ...], ...],
    values: tuple[tuple[Scalar, ...], ...],
) -> tuple[tuple[Scalar, ...], ...]:
    """Solve all RHS columns in one full-column-rank basis matrix."""
    width = len(basis[0]) if basis else 0
    if width == 0:
        if any(value != 0 for row in values for value in row):
            raise _domain(
                "restriction_not_preserved",
                "target restriction leaves the pointwise image",
            )
        return ()
    augmented = [[*basis[row], *values[row]] for row in range(len(basis))]
    reduced, pivots = _cochain_rref(field, augmented)
    if any(pivot >= width for pivot in pivots):
        raise _domain(
            "restriction_not_preserved", "target restriction leaves the pointwise image"
        )
    right_width = len(values[0]) if values else 0
    coordinates = [[field.zero() for _ in range(right_width)] for _ in range(width)]
    for row, pivot in enumerate(pivots):
        if pivot < width:
            for column in range(right_width):
                coordinates[pivot][column] = reduced[row][width + column]
    return tuple(tuple(row) for row in coordinates)


def image_of_morphism(value: SheafMorphismResult) -> SheafMorphismImageResult:
    """Compute the image sheaf and exact factorization ``F -> im(phi) -> G``."""
    source, target = value.source, value.target
    field = _admit_field(source.coefficient_field, source.prime)
    _admit_field(target.coefficient_field, target.prime)
    if (
        source.complex != target.complex
        or source.coefficient_field != target.coefficient_field
        or source.prime != target.prime
    ):
        raise _domain(
            "parent_mismatch",
            "sheaf morphisms require one complex and coefficient field",
        )
    _admit_section_plan(source)
    _admit_section_plan(target)
    if not isinstance(value.components, tuple) or any(
        not isinstance(component, tuple)
        or len(component) != 2
        or not isinstance(component[1], (tuple, list))
        for component in value.components
    ):
        raise _domain(
            "component_structure", "morphism components must be (simplex, matrix) pairs"
        )
    target_cover, input_digits, morphism_work = _admit_morphism_resources(
        source, target, value.components
    )
    cells = source.canonical_face_order
    keys = tuple(
        _resolve_component_key(key, cells) for key, _matrix in value.components
    )
    if keys != cells:
        raise _domain(
            "component_axis",
            "one morphism component per simplex in canonical order is required",
        )
    source_stalks = {stalk.simplex: stalk for stalk in source.stalks}
    target_stalks = {stalk.simplex: stalk for stalk in target.stalks}
    components: dict[tuple[str, ...], tuple[tuple[Scalar, ...], ...]] = {}
    canonical_components: list[Component] = []
    for cell, (_key, raw) in zip(cells, value.components, strict=True):
        matrix = _admit_component_matrix(raw, field, ("components", ".".join(cell)))
        if len(matrix) != len(target_stalks[cell].basis) or any(
            len(row) != len(source_stalks[cell].basis) for row in matrix
        ):
            raise _domain(
                "component_shape", "morphism components must match their stalk axes"
            )
        components[cell] = matrix
        canonical_components.append((cell, field.render(matrix)))

    target_restrictions = {
        (item.source, item.target): item
        for item in (*target.cover_restrictions, *target.derived_restrictions)
    }
    restrictions = {
        (item.source, item.target): item
        for item in (*source.cover_restrictions, *source.derived_restrictions)
    }
    # A uniform conservative bound covers local RREF plus every induced map solve.
    work = morphism_work
    for cell in cells:
        target_rank = len(target_stalks[cell].basis)
        source_rank = len(source_stalks[cell].basis)
        work += max(1, target_rank**2 * (target_rank + source_rank))
    for item in restrictions.values():
        left, right = (
            len(target_stalks[item.source].basis),
            len(target_stalks[item.target].basis),
        )
        induced_rank_bound = min(left, right, MAX_SHEAF_STALK_RANK)
        work += max(1, right * induced_rank_bound * left)
        work += max(1, 2 * right**2 * (2 * induced_rank_bound))
    reconstruction_work, reconstruction_chars = _parent_reconstruction_bounds(
        (source, target), input_digits=input_digits
    )
    if work + reconstruction_work > MAX_SHEAF_MORPHISM_WORK:
        raise _resource(
            "work_bound",
            "naturality, parent reconstruction, and image work exceed the admitted bound",
        )
    max_rank = min(
        MAX_SHEAF_STALK_RANK,
        max((len(stalk.basis) for stalk in target.stalks), default=0),
    )
    rank_growth = max(
        1,
        max((len(stalk.basis) for stalk in source.stalks), default=0),
        max((len(stalk.basis) for stalk in target.stalks), default=0),
    )
    restriction_cells = sum(
        len(target_stalks[a].basis) * len(target_stalks[b].basis)
        for a, b in restrictions
    )
    output_cells = (
        restriction_cells
        + len(cells) * max_rank * MAX_SHEAF_STALK_RANK
        + sum(len(stalk.basis) * max_rank for stalk in source.stalks)
    )
    if source.coefficient_field is SheafField.RATIONAL:
        output_digits = 2 * rank_growth**2 * input_digits + 2 * rank_growth**2 + 32
    else:
        output_digits = max(input_digits, len(str(source.prime)))
    if (
        4 * len(value.model_dump_json())
        + reconstruction_chars
        + sheaf_scalar_json_bound(output_cells, output_digits)
        > MAX_SHEAF_MORPHISM_OUTPUT_CHARS
    ):
        raise _resource(
            "output_bound",
            "image restrictions and factorization exceed the output envelope",
        )

    _readmit_parent_sheaf(source, role="source")
    _readmit_parent_sheaf(target, role="target")
    # Re-establish naturality from the cover squares before using its image
    # preservation consequence. The complete combined arithmetic envelope has
    # been admitted and both parent diagrams have now been reconstructed.
    for item in source.cover_restrictions:
        source_restriction = tuple(
            tuple(field.parse(value) for value in row) for row in item.entries
        )
        target_item = target_cover[(item.source, item.target)]
        target_restriction = tuple(
            tuple(field.parse(value) for value in row) for row in target_item.entries
        )
        left = _mul(
            [list(row) for row in components[item.target]],
            [list(row) for row in source_restriction],
            field.prime,
            output_width=len(source_stalks[item.source].basis),
        )
        right = _mul(
            [list(row) for row in target_restriction],
            [list(row) for row in components[item.source]],
            field.prime,
            output_width=len(source_stalks[item.source].basis),
        )
        if left != right:
            raise _domain(
                "morphism_not_natural", "image requires a natural sheaf morphism"
            )
    checked = SheafMorphismResult(
        source=source,
        target=target,
        components=tuple(canonical_components),
        natural=True,
    )

    bases: dict[tuple[str, ...], tuple[tuple[Scalar, ...], ...]] = {}
    ranks: dict[tuple[str, ...], int] = {}
    image_stalks: list[SheafStalk] = []
    inclusions: list[Component] = []
    factors: list[Component] = []
    for cell in cells:
        target_rank = len(target_stalks[cell].basis)
        source_rank = len(source_stalks[cell].basis)
        matrix = components[cell]
        pivots = _independent_columns(field, matrix, source_rank)
        basis = tuple(
            tuple(matrix[row][column] for column in pivots)
            for row in range(target_rank)
        )
        bases[cell] = basis
        rank = len(pivots)
        ranks[cell] = rank
        labels = tuple(f"i{index}" for index in range(rank))
        image_stalks.append(SheafStalk(simplex=cell, basis=labels))
        inclusions.append((cell, field.render(basis)))
        factor_matrix = _solve_matrix(field, basis, matrix)
        factors.append((cell, field.render(factor_matrix)))

    def induced(
        item: SheafRestriction, target_item: SheafRestriction
    ) -> SheafRestriction:
        target_matrix = tuple(
            tuple(field.parse(x) for x in row) for row in target_item.entries
        )
        source_basis = bases[item.source]
        image = _mul(
            [list(row) for row in target_matrix],
            [list(row) for row in source_basis],
            field.prime,
            output_width=ranks[item.source],
        )
        values = tuple(
            tuple(image[row][column] for column in range(ranks[item.source]))
            for row in range(len(image))
        )
        entries = _solve_matrix(field, bases[item.target], values)
        return SheafRestriction(
            source=item.source,
            target=item.target,
            row_basis=tuple(f"i{i}" for i in range(ranks[item.target])),
            column_basis=tuple(f"i{i}" for i in range(ranks[item.source])),
            entries=field.render(entries),
            cover_path=item.cover_path,
        )

    image_sheaf = FiniteCellularSheaf._from_kernel(
        complex=source.complex,
        coefficient_field=source.coefficient_field,
        prime=source.prime,
        stalks=tuple(image_stalks),
        cover_restrictions=tuple(
            induced(item, target_restrictions[(item.source, item.target)])
            for item in source.cover_restrictions
        ),
        derived_restrictions=tuple(
            induced(item, target_restrictions[(item.source, item.target)])
            for item in source.derived_restrictions
        ),
        diamonds=source.diamonds,
        comparable_pairs=source.comparable_pairs,
    )
    inclusion = SheafMorphismResult(
        source=image_sheaf, target=target, components=tuple(inclusions), natural=True
    )
    factor = SheafMorphismResult(
        source=source, target=image_sheaf, components=tuple(factors), natural=True
    )
    return SheafMorphismImageResult._from_image(
        morphism=checked, image=image_sheaf, inclusion=inclusion, factor=factor
    )


__all__ = ["SheafMorphismImageRequest", "SheafMorphismImageResult", "image_of_morphism"]
