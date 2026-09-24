"""Exact pointwise kernels of finite cellular sheaf morphisms."""

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
    _cochain_nullspace,
    _cochain_rref,
    _ExactField,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_MORPHISM_OUTPUT_CHARS,
    MAX_SHEAF_MORPHISM_WORK,
    MAX_SHEAF_STALK_RANK,
    FiniteCellularSheaf,
    SheafRestriction,
    SheafStalk,
    sheaf_scalar_json_bound,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    Component,
    SheafMorphismResult,
    _complete_diagram,
    _mul,
    _scan_morphism_scalar_text,
    morphism,
)


class SheafMorphismKernelRequest(StrictModel):
    """A candidate stalkwise map whose natural kernel sheaf is requested."""

    morphism: SheafMorphismResult


class SheafMorphismKernelResult(StrictModel):
    """Kernel sheaf and its canonical inclusion into the morphism source."""

    morphism: SheafMorphismResult
    kernel: FiniteCellularSheaf
    inclusion: SheafMorphismResult

    @model_validator(mode="after")
    def bind_kernel_inclusion(self) -> Self:
        source = self.morphism.source
        cells = source.canonical_face_order
        if self.kernel.complex != self.morphism.source.complex:
            raise ValueError("kernel must retain the source complex")
        if (
            self.kernel.coefficient_field != self.morphism.source.coefficient_field
            or self.kernel.prime != self.morphism.source.prime
        ):
            raise ValueError("kernel must retain the source coefficient field")
        if (
            self.inclusion.source != self.kernel
            or self.inclusion.target != self.morphism.source
        ):
            raise ValueError("inclusion must map the kernel sheaf into the source")
        if tuple(stalk.simplex for stalk in self.kernel.stalks) != cells:
            raise ValueError("kernel stalks must retain the canonical source axes")
        if not self.inclusion.natural or self.inclusion.obstruction is not None:
            raise ValueError("kernel inclusion must be declared natural")
        if tuple(key for key, _ in self.inclusion.components) != cells:
            raise ValueError(
                "inclusion components must retain the canonical source axes"
            )
        source_ranks = {stalk.simplex: len(stalk.basis) for stalk in source.stalks}
        kernel_ranks = {stalk.simplex: len(stalk.basis) for stalk in self.kernel.stalks}
        for component_key, matrix in self.inclusion.components:
            if isinstance(component_key, str):
                raise ValueError("inclusion components must use tuple simplex axes")
            cell = tuple(component_key)
            if len(matrix) != source_ranks[cell] or any(
                len(row) != kernel_ranks[cell] for row in matrix
            ):
                raise ValueError(
                    "inclusion component axes must match source and kernel stalks"
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        morphism: SheafMorphismResult,
        kernel: FiniteCellularSheaf,
        inclusion: SheafMorphismResult,
    ) -> Self:
        """Build the admitted producer result without replaying its relation."""
        return cls.model_construct(
            morphism=morphism,
            kernel=kernel,
            inclusion=inclusion,
        )


def _fail_domain(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("morphism",),
        code=f"topology.cellular_sheaf.morphism_kernel.{code}",
        message=message,
    )


def _fail_resource(code: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("morphism",),
        code=f"topology.cellular_sheaf.morphism_kernel.{code}",
        message=message,
    )


def _coordinates(
    field: _ExactField,
    basis_rows: list[list[Scalar]],
    vector: list[Scalar],
) -> list[Scalar]:
    """Solve Bx=v for independent basis columns, retaining exact coordinates."""
    width = len(basis_rows[0]) if basis_rows else 0
    if width == 0:
        if any(value != 0 for value in vector):
            raise _fail_domain(
                "restriction_not_preserved",
                "source restriction does not preserve the pointwise kernel",
            )
        return []
    rows = [[*basis_rows[row], vector[row]] for row in range(len(vector))]
    reduced, pivots = _cochain_rref(field, rows)
    if width in pivots:
        raise _fail_domain(
            "restriction_not_preserved",
            "source restriction does not preserve the pointwise kernel",
        )
    values = [field.zero() for _ in range(width)]
    for row, pivot in enumerate(pivots):
        if pivot < width:
            values[pivot] = reduced[row][-1]
    return values


def kernel_of_morphism(value: SheafMorphismResult) -> SheafMorphismKernelResult:
    """Compute the categorical kernel in finite-dimensional based stalks."""
    source, target = value.source, value.target
    field = _admit_field(source.coefficient_field, source.prime)
    # Re-run the naturality and parent checks. A serialized natural flag is not evidence.
    checked = morphism(source, target, value.components)
    if not checked.natural:
        raise _fail_domain(
            "morphism_not_natural", "kernel requires a natural sheaf morphism"
        )
    _complete_diagram(source, role="source")
    _complete_diagram(target, role="target")
    cells = source.canonical_face_order
    source_stalks = {stalk.simplex: stalk for stalk in source.stalks}
    target_stalks = {stalk.simplex: stalk for stalk in target.stalks}
    components = {
        key: tuple(tuple(field.parse(x) for x in row) for row in matrix)
        for key, matrix in checked.components
    }
    restrictions = {
        (item.source, item.target): item
        for item in (*source.cover_restrictions, *source.derived_restrictions)
    }
    work = 0
    for cell in cells:
        f_rank = len(source_stalks[cell].basis)
        g_rank = len(target_stalks[cell].basis)
        work += max(1, f_rank) ** 3 + g_rank * f_rank * max(1, f_rank)
    restriction_cells = sum(
        len(source_stalks[a].basis) * len(source_stalks[b].basis)
        for a, b in restrictions
    )
    work += restriction_cells * max(1, MAX_SHEAF_STALK_RANK)
    if work > MAX_SHEAF_MORPHISM_WORK:
        raise _fail_resource(
            "work_bound",
            "pointwise kernel and induced restriction work exceeds its bound",
        )
    # Input sheaves/morphism were already bounded by their owners. Bound the
    # worst-case output before nullspace or restriction expansion.
    output_cells = (
        sum(
            len(source_stalks[cell].basis) * MAX_SHEAF_STALK_RANK
            + MAX_SHEAF_STALK_RANK * MAX_SHEAF_STALK_RANK
            for cell in cells
        )
        + restriction_cells * MAX_SHEAF_STALK_RANK
    )
    _, input_digits = _scan_morphism_scalar_text(source, target, checked.components)
    output_digits = 2 * MAX_SHEAF_STALK_RANK * input_digits + 32
    if (
        4 * len(checked.model_dump_json())
        + sheaf_scalar_json_bound(output_cells, output_digits)
        > MAX_SHEAF_MORPHISM_OUTPUT_CHARS
    ):
        raise _fail_resource(
            "output_bound",
            "kernel basis, restrictions, and inclusion exceed their output envelope",
        )

    bases: dict[tuple[str, ...], list[list[Scalar]]] = {}
    kernel_ranks: dict[tuple[str, ...], int] = {}
    kernel_stalks = []
    inclusion_components: list[Component] = []
    for cell in cells:
        rank = len(source_stalks[cell].basis)
        null_vectors = _cochain_nullspace(
            field, [list(row) for row in components[cell]], rank
        )
        # Nullspace helper returns vectors as rows; transpose to inclusion matrix.
        columns = [[vector[row] for vector in null_vectors] for row in range(rank)]
        bases[cell] = columns
        kernel_ranks[cell] = len(null_vectors)
        labels = tuple(f"k{i}" for i in range(len(null_vectors)))
        kernel_stalks.append(SheafStalk(simplex=cell, basis=labels))
        inclusion_components.append(
            (cell, field.render(tuple(tuple(row) for row in columns)))
        )

    def induced(item: SheafRestriction) -> SheafRestriction:
        source_basis = bases[item.source]
        target_basis = bases[item.target]
        restriction_matrix = [[field.parse(x) for x in row] for row in item.entries]
        image = _mul(
            restriction_matrix,
            source_basis,
            field.prime,
            output_width=kernel_ranks[item.source],
        )
        image_cols = [
            [image[row][col] for row in range(len(image))]
            for col in range(len(image[0]) if image else 0)
        ]
        coordinate_cols = [
            _coordinates(
                field, target_basis, [image[row][col] for row in range(len(image))]
            )
            for col in range(len(image_cols))
        ]
        entries = [
            [coordinate_cols[col][row] for col in range(len(coordinate_cols))]
            for row in range(kernel_ranks[item.target])
        ]
        return SheafRestriction(
            source=item.source,
            target=item.target,
            row_basis=tuple(f"k{i}" for i in range(kernel_ranks[item.target])),
            column_basis=tuple(f"k{i}" for i in range(kernel_ranks[item.source])),
            entries=field.render(tuple(tuple(row) for row in entries)),
            cover_path=item.cover_path,
        )

    covers = tuple(induced(item) for item in source.cover_restrictions)
    derived = tuple(induced(item) for item in source.derived_restrictions)
    kernel = FiniteCellularSheaf._from_kernel(
        complex=source.complex,
        coefficient_field=source.coefficient_field,
        prime=source.prime,
        stalks=tuple(kernel_stalks),
        cover_restrictions=covers,
        derived_restrictions=derived,
        diamonds=source.diamonds,
        comparable_pairs=source.comparable_pairs,
    )
    # Naturality follows from the coordinate solves above: each source
    # restriction applied to the inclusion columns is exactly the target
    # inclusion matrix times the returned kernel restriction.
    inclusion = SheafMorphismResult(
        source=kernel,
        target=source,
        components=tuple(inclusion_components),
        natural=True,
    )
    return SheafMorphismKernelResult._from_kernel(
        morphism=checked,
        kernel=kernel,
        inclusion=inclusion,
    )


__all__ = [
    "SheafMorphismKernelRequest",
    "SheafMorphismKernelResult",
    "kernel_of_morphism",
]
