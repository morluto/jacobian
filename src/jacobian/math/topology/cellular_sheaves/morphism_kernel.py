"""Exact pointwise kernels of finite cellular sheaf morphisms."""

from __future__ import annotations

from itertools import pairwise
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
    from_cover_maps,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_MORPHISM_OUTPUT_CHARS,
    MAX_SHEAF_MORPHISM_WORK,
    MAX_SHEAF_STALK_RANK,
    CoverRestrictionMatrix,
    FiniteCellularSheaf,
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


def _readmit_parent_sheaf(
    sheaf: FiniteCellularSheaf,
    *,
    role: str,
) -> None:
    """Reconstruct the parent diagram from covers before relying on derived maps."""
    cover_maps = tuple(
        CoverRestrictionMatrix(
            source=restriction.source,
            target=restriction.target,
            entries=restriction.entries,
        )
        for restriction in sheaf.cover_restrictions
    )
    admitted = from_cover_maps(
        sheaf.complex,
        sheaf.coefficient_field,
        sheaf.prime,
        sheaf.stalks,
        cover_maps,
    )
    if admitted.sheaf is None or admitted.sheaf != sheaf:
        raise _fail_domain(
            "parent_diagram_not_admitted",
            f"the {role} sheaf restrictions must equal the exact functor diagram reconstructed from its cover maps",
        )


def _parent_reconstruction_bounds(
    parents: tuple[FiniteCellularSheaf, FiniteCellularSheaf],
    *,
    input_digits: int,
) -> tuple[int, int]:
    """Bound combined exact cover coherence reconstruction before either run."""
    total_work = 0
    output_cells = 0
    output_digits = 1
    for sheaf in parents:
        cells = sheaf.canonical_face_order
        ranks = {stalk.simplex: len(stalk.basis) for stalk in sheaf.stalks}
        for source in cells:
            for target in cells:
                gap = len(target) - len(source)
                if gap < 2 or not set(source).issubset(target):
                    continue
                chain = [source]
                current = source
                for vertex in sorted(set(target) - set(source)):
                    current = tuple(sorted((*current, vertex)))
                    chain.append(current)
                for earlier, later in pairwise(chain):
                    total_work += max(
                        1,
                        ranks[source] * ranks[earlier] * ranks[later],
                    )
                output_cells += ranks[source] * ranks[target]
                path_rank = max((ranks[cell] for cell in chain), default=0)
                if path_rank:
                    # A path of m matrix products has at most r^(m-1) terms
                    # per entry. This bounds rational numerator/denominator
                    # growth before the exact composites are materialized.
                    term_count = path_rank ** (gap - 1)
                    path_digits = (
                        gap * input_digits * term_count + len(str(term_count)) + 2
                    )
                    output_digits = max(output_digits, path_digits)

        # The cover verifier compares all pairs of saturated paths in every
        # length-two diamond. Count matrix scalar products before calling it.
        for source in cells:
            for target in cells:
                if len(target) - len(source) != 2 or not set(source).issubset(target):
                    continue
                middles = tuple(
                    middle
                    for middle in cells
                    if len(middle) == len(source) + 1
                    and set(source) < set(middle) < set(target)
                )
                for left_index, left_middle in enumerate(middles):
                    for right_middle in middles[left_index + 1 :]:
                        total_work += 2 * max(
                            1,
                            ranks[source] * ranks[left_middle] * ranks[target],
                            ranks[source] * ranks[right_middle] * ranks[target],
                        )
    return total_work, sheaf_scalar_json_bound(output_cells, output_digits)


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
    if type(value) is not SheafMorphismResult:
        raise _fail_domain("morphism_type", "morphism must be a canonical sheaf map")
    if type(value.source) is not FiniteCellularSheaf or type(value.target) is not FiniteCellularSheaf:
        raise _fail_domain("parent_type", "morphism endpoints must be cellular sheaves")
    source, target = value.source, value.target
    if not isinstance(value.components, tuple) or any(
        not isinstance(component, tuple)
        or len(component) != 2
        or not isinstance(component[1], (tuple, list))
        or not (
            isinstance(component[0], str)
            or (
                isinstance(component[0], tuple)
                and all(isinstance(label, str) for label in component[0])
            )
        )
        for component in value.components
    ):
        raise _fail_domain(
            "component_structure",
            "morphism components must be keyed (simplex, matrix) pairs",
        )
    field = _admit_field(source.coefficient_field, source.prime)
    target_field = _admit_field(target.coefficient_field, target.prime)
    if (
        source.complex != target.complex
        or source.coefficient_field != target.coefficient_field
        or source.prime != target.prime
    ):
        raise _fail_domain(
            "parent_mismatch",
            "sheaf morphisms require one complex and coefficient field",
        )
    # Authored Pydantic models can bypass validation via model_construct/model_copy;
    # reject malformed pair structure before the shared scalar/resource scan.
    target_cover, input_digits, morphism_work = _admit_morphism_resources(
        source, target, value.components
    )
    cells = source.canonical_face_order
    normalized_keys = tuple(
        _resolve_component_key(key, cells) for key, _matrix in value.components
    )
    if normalized_keys != cells:
        raise _fail_domain(
            "component_axis",
            "one morphism component per simplex in canonical order is required",
        )
    source_stalks = {stalk.simplex: stalk for stalk in source.stalks}
    target_stalks = {stalk.simplex: stalk for stalk in target.stalks}
    components: dict[tuple[str, ...], tuple[tuple[Scalar, ...], ...]] = {}
    canonical_components: list[Component] = []
    for cell, (_key, matrix) in zip(cells, value.components, strict=True):
        parsed = _admit_component_matrix(matrix, field, ("components", ".".join(cell)))
        if len(parsed) != len(target_stalks[cell].basis) or any(
            len(row) != len(source_stalks[cell].basis) for row in parsed
        ):
            raise _fail_domain(
                "component_shape",
                "morphism component matrices must match their stalk axes",
            )
        components[cell] = parsed
        canonical_components.append((cell, field.render(parsed)))
    restrictions = {
        (item.source, item.target): item
        for item in (*source.cover_restrictions, *source.derived_restrictions)
    }
    kernel_work = 0
    for cell in cells:
        f_rank = len(source_stalks[cell].basis)
        g_rank = len(target_stalks[cell].basis)
        kernel_work += max(1, f_rank) ** 3 + g_rank * f_rank * max(1, f_rank)
    restriction_cells = sum(
        len(source_stalks[a].basis) * len(source_stalks[b].basis)
        for a, b in restrictions
    )
    kernel_work += restriction_cells * max(1, MAX_SHEAF_STALK_RANK)
    reconstruction_work, reconstruction_chars = _parent_reconstruction_bounds(
        (source, target), input_digits=input_digits
    )
    total_work = kernel_work + morphism_work + reconstruction_work
    if total_work > MAX_SHEAF_MORPHISM_WORK:
        raise _fail_resource(
            "work_bound",
            "parent coherence, naturality, and kernel work exceed the combined bound",
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
    output_digits = 2 * MAX_SHEAF_STALK_RANK * input_digits + 32
    if (
        4 * len(value.model_dump_json())
        + reconstruction_chars
        + sheaf_scalar_json_bound(output_cells, output_digits)
        > MAX_SHEAF_MORPHISM_OUTPUT_CHARS
    ):
        raise _fail_resource(
            "output_bound",
            "kernel basis, restrictions, and inclusion exceed their output envelope",
        )

    # Reconstruct the parent diagrams only after the combined admission above;
    # reconstruction itself expands exact matrices and must be included in the
    # same resource envelope as kernel construction.
    _readmit_parent_sheaf(source, role="source")
    _readmit_parent_sheaf(target, role="target")

    # Reconstruct both complete parent functors only after the combined input,
    # arithmetic, and output envelope has been admitted.

    # Do not trust the serialized natural flag until both parent diagrams have
    # been reconstructed and shown equal to their cover-map presentations.
    for restriction in source.cover_restrictions:
        source_component = components[restriction.source]
        target_component = components[restriction.target]
        source_restriction = tuple(
            tuple(field.parse(entry) for entry in row) for row in restriction.entries
        )
        target_restriction = tuple(
            tuple(target_field.parse(entry) for entry in row)
            for row in target_cover[(restriction.source, restriction.target)].entries
        )
        left = _mul(
            [list(row) for row in target_component],
            [list(row) for row in source_restriction],
            field.prime,
            output_width=len(source_stalks[restriction.source].basis),
        )
        right = _mul(
            [list(row) for row in target_restriction],
            [list(row) for row in source_component],
            field.prime,
            output_width=len(source_stalks[restriction.source].basis),
        )
        if left != right:
            raise _fail_domain(
                "morphism_not_natural", "kernel requires a natural sheaf morphism"
            )
    checked = SheafMorphismResult(
        source=source,
        target=target,
        components=tuple(canonical_components),
        natural=True,
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
