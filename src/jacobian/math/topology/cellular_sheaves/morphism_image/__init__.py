"""Exact pointwise images of finite cellular sheaf morphisms."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ValidationError, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    FacesInDimension,
    FiniteSimplicialComplex,
)
from jacobian.math.topology.cellular_sheaves._kernel import (
    Scalar,
    _admit_field,
    _canonical_chain,
    _cochain_rref,
    _ExactField,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_COVER_MAPS,
    MAX_SHEAF_DERIVED_RESTRICTIONS,
    MAX_SHEAF_MORPHISM_OUTPUT_DIGIT_WORK,
    MAX_SHEAF_MORPHISM_PARENT_CELLS,
    MAX_SHEAF_MORPHISM_WORK,
    MAX_SHEAF_SIMPLICES,
    MAX_SHEAF_STALK_RANK,
    FiniteCellularSheaf,
    SheafField,
    SheafRestriction,
    SheafStalk,
    sheaf_scalar_digit_work,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    Component,
    SheafMorphismResult,
    _admit_component_matrix,
    _admit_morphism_resources,
    _mul,
    _resolve_component_key,
)
from jacobian.math.topology.cellular_sheaves.morphism_kernel import (
    _parent_reconstruction_bounds,
    _readmit_parent_sheaf,
)

# The exact canonical model types reachable in a sheaf payload. Any other model
# type, including a subclass, may declare fields this walk must not expand.
_PAYLOAD_MODEL_TYPES = frozenset(
    {
        CanonicalRational,
        FacesInDimension,
        FiniteCellularSheaf,
        FiniteSimplicialComplex,
        SheafRestriction,
        SheafStalk,
    }
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
        if target.complex != source.complex:
            raise ValueError("morphism source and target must use the same complex")
        if tuple(stalk.simplex for stalk in self.image.stalks) != cells:
            raise ValueError("image stalks must retain the canonical simplex axes")
        source_ranks = {stalk.simplex: len(stalk.basis) for stalk in source.stalks}
        target_ranks = {stalk.simplex: len(stalk.basis) for stalk in target.stalks}
        image_ranks = {stalk.simplex: len(stalk.basis) for stalk in self.image.stalks}
        _require_image_diagram_axes(self.image, source, target)
        for map_ in (self.morphism, self.inclusion, self.factor):
            if not map_.natural or map_.obstruction is not None:
                raise ValueError("image factorization morphisms must be natural")
        _require_component_axes(
            self.morphism.components, cells, target_ranks, source_ranks, "morphism"
        )
        _require_component_axes(
            self.inclusion.components, cells, target_ranks, image_ranks, "inclusion"
        )
        _require_component_axes(
            self.factor.components, cells, image_ranks, source_ranks, "factor"
        )
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


def _bounded_restriction_sequence(
    value: object, limit: int, role: str, field: str
) -> tuple[object, ...] | list[object]:
    if not isinstance(value, (tuple, list)) or type(value) not in (tuple, list):
        raise _domain(
            "parent_restriction_structure",
            f"{role} restriction {field} must be an ordered container",
        )
    if len(value) > limit:
        raise _resource(
            "parent_restriction_shape",
            f"{role} restriction {field} exceeds its {limit}-element envelope",
        )
    return value


def _restriction_labels(value: object, limit: int, role: str, field: str) -> None:
    labels = _bounded_restriction_sequence(value, limit, role, field)
    if any(not isinstance(label, str) for label in labels):
        raise _domain(
            "parent_restriction_labels",
            f"{role} restriction {field} must contain scalar labels",
        )


def _reject_oversized_restriction(restriction: object, role: str) -> None:
    """Bound every nested matrix axis before the parent is recursively rebuilt."""
    if type(restriction) is not SheafRestriction:
        raise _domain(
            "parent_restriction_type",
            f"{role} restrictions must be typed restriction matrices",
        )
    for field in ("row_basis", "column_basis"):
        _restriction_labels(
            getattr(restriction, field, None), MAX_SHEAF_STALK_RANK, role, field
        )
    rows = _bounded_restriction_sequence(
        getattr(restriction, "entries", None), MAX_SHEAF_STALK_RANK, role, "entries"
    )
    for row in rows:
        entries = _bounded_restriction_sequence(row, MAX_SHEAF_STALK_RANK, role, "row")
        for scalar in entries:
            if type(scalar) is CanonicalRational:
                valid = (
                    type(getattr(scalar, "num", None)) is int
                    and type(getattr(scalar, "den", None)) is int
                )
            else:
                valid = type(scalar) is int
            if not valid:
                raise _domain(
                    "parent_restriction_scalar",
                    f"{role} restriction entries must contain exact scalar values",
                )
    for field in ("source", "target"):
        _restriction_labels(
            getattr(restriction, field, None), MAX_SHEAF_SIMPLICES, role, field
        )
    path = _bounded_restriction_sequence(
        getattr(restriction, "cover_path", None),
        MAX_SHEAF_SIMPLICES,
        role,
        "cover_path",
    )
    for simplex in path:
        _restriction_labels(simplex, MAX_SHEAF_SIMPLICES, role, "cover_path simplex")


def _reject_malformed_parent_complex(value: object, role: str) -> None:
    """Bound the retained face containers before copying any canonical complex."""
    if type(value) is not FiniteSimplicialComplex:
        raise _domain(
            "parent_complex_type", f"{role} parent must retain a typed complex"
        )
    _restriction_labels(
        getattr(value, "vertices", None), MAX_SHEAF_SIMPLICES, role, "complex vertices"
    )
    facets = _bounded_restriction_sequence(
        getattr(value, "maximal_simplices", None),
        MAX_SHEAF_SIMPLICES,
        role,
        "complex facets",
    )
    for facet in facets:
        _restriction_labels(facet, MAX_TOPOLOGY_DIMENSION + 1, role, "complex facet")
    groups = _bounded_restriction_sequence(
        getattr(value, "faces_by_dimension", None),
        MAX_TOPOLOGY_DIMENSION + 1,
        role,
        "complex dimensions",
    )
    face_count = 0
    for group in groups:
        if (
            type(group) is not FacesInDimension
            or type(getattr(group, "dimension", None)) is not int
        ):
            raise _domain(
                "parent_complex_structure", f"{role} parent face groups must be typed"
            )
        faces = _bounded_restriction_sequence(
            getattr(group, "faces", None), MAX_SHEAF_SIMPLICES, role, "complex faces"
        )
        face_count += len(faces)
        if face_count > MAX_SHEAF_SIMPLICES:
            raise _resource(
                "parent_simpices", f"{role} parent exceeds the simplex envelope"
            )
        for face in faces:
            _restriction_labels(
                face, MAX_TOPOLOGY_DIMENSION + 1, role, "complex simplex"
            )
    counts = _bounded_restriction_sequence(
        getattr(value, "f_vector", None),
        MAX_TOPOLOGY_DIMENSION + 1,
        role,
        "complex face counts",
    )
    if any(type(count) is not int for count in counts) or any(
        type(getattr(value, field, None)) is not int
        for field in ("dimension", "closure_size")
    ):
        raise _domain(
            "parent_complex_structure", f"{role} parent face counts must be integers"
        )
    if (
        not isinstance(getattr(value, "orientation_convention", None), str)
        or type(getattr(value, "empty_simplex_stored", None)) is not bool
    ):
        raise _domain(
            "parent_complex_structure",
            f"{role} parent complex conventions must be scalar values",
        )


def _reject_malformed_stalk(stalk: object, role: str) -> None:
    if type(stalk) is not SheafStalk:
        raise _domain("parent_stalk_type", f"{role} parent stalks must be typed")
    basis = getattr(stalk, "basis", None)
    if (
        isinstance(basis, (tuple, list))
        and type(basis) in (tuple, list)
        and len(basis) > MAX_SHEAF_STALK_RANK
    ):
        raise _resource(
            "parent_stalk_rank",
            f"{role} morphism parent retains a stalk above the rank envelope",
        )
    _restriction_labels(basis, MAX_SHEAF_STALK_RANK, role, "stalk basis")
    _restriction_labels(
        getattr(stalk, "simplex", None),
        MAX_TOPOLOGY_DIMENSION + 1,
        role,
        "stalk simplex",
    )


def _reject_oversized_parent_container(parent: object, role: str) -> None:
    """Refuse an over-envelope parent container before any recursive copy.

    ``MAX_SHEAF_SIMPLICES`` bounds the simplices a canonical cellular sheaf may
    retain, and each stalk basis and restriction row is separately bounded. A
    ``model_construct`` or ``model_copy`` value bypasses those declared bounds,
    so check the container lengths here rather than discovering them after the
    payload has been rebuilt.
    """
    stalks = getattr(parent, "stalks", None)
    if not isinstance(stalks, (tuple, list)) or type(stalks) not in (tuple, list):
        raise _domain(
            "parent_container_type", f"{role} stalks must be an ordered container"
        )
    if len(stalks) > MAX_SHEAF_SIMPLICES:
        raise _resource(
            "parent_simpices",
            f"{role} morphism parent retains more than {MAX_SHEAF_SIMPLICES} simplices",
        )
    for attribute, limit in (
        ("cover_restrictions", MAX_SHEAF_COVER_MAPS),
        ("derived_restrictions", MAX_SHEAF_DERIVED_RESTRICTIONS),
    ):
        container = getattr(parent, attribute, None)
        if not isinstance(container, (tuple, list)) or type(container) not in (
            tuple,
            list,
        ):
            raise _domain(
                "parent_container_type",
                f"{role} {attribute} must be an ordered container",
            )
        if len(container) > limit:
            raise _resource(
                "parent_cells",
                f"{role} morphism parent retains too many {attribute} rows",
            )
        for restriction in container:
            _reject_oversized_restriction(restriction, role)
    for stalk in stalks:
        _reject_malformed_stalk(stalk, role)
    _reject_malformed_parent_complex(getattr(parent, "complex", None), role)
    if not isinstance(getattr(parent, "coefficient_field", None), SheafField):
        raise _domain(
            "parent_field_type", f"{role} parent must retain a typed coefficient field"
        )
    prime = getattr(parent, "prime", None)
    if prime is not None and type(prime) is not int:
        raise _domain("parent_field_type", f"{role} parent prime must be an integer")
    if any(
        type(getattr(parent, field, None)) is not int
        for field in ("diamonds", "comparable_pairs")
    ):
        raise _domain(
            "parent_count_type", f"{role} parent incidence counts must be integers"
        )


def _unvalidated_payload(value: object) -> object:
    """Expose nested model fields as raw containers for trust-boundary revalidation."""
    if isinstance(value, BaseModel):
        # Require the exact canonical model type. A subclass legitimately
        # declares extra fields, so type(value).model_fields would treat them as
        # declared and copy their whole unbounded payload before base-model
        # validation rejects them as extra data.
        if type(value) not in _PAYLOAD_MODEL_TYPES:
            raise _domain(
                "parent_structure",
                "parent models must be canonical types retaining only declared fields",
            )
        fields = type(value).model_fields
        if len(value.__dict__) != len(fields) or any(
            key not in fields for key in value.__dict__
        ):
            raise _domain(
                "parent_structure",
                "parent models must retain exactly their declared fields",
            )
        return {key: _unvalidated_payload(item) for key, item in value.__dict__.items()}
    if type(value) is tuple:
        return tuple(_unvalidated_payload(item) for item in value)
    if type(value) is list:
        return [_unvalidated_payload(item) for item in value]
    if isinstance(value, (tuple, list, dict)):
        raise _domain(
            "parent_structure", "parent fields must use canonical builtin containers"
        )
    return value


def _require_image_diagram_axes(
    image: FiniteCellularSheaf,
    source: FiniteCellularSheaf,
    target: FiniteCellularSheaf,
) -> None:
    cells = source.canonical_face_order
    expected_cover_axes = tuple(
        sorted(
            (earlier, later, (earlier, later))
            for earlier in cells
            for later in cells
            if len(later) == len(earlier) + 1 and set(earlier) < set(later)
        )
    )
    expected_derived_axes = tuple(
        sorted(
            (earlier, later, tuple(_canonical_chain(earlier, later)))
            for earlier in cells
            for later in cells
            if set(earlier) < set(later) and len(later) > len(earlier) + 1
        )
    )
    for kind in ("cover_restrictions", "derived_restrictions"):
        expected_axes = (
            expected_cover_axes
            if kind == "cover_restrictions"
            else expected_derived_axes
        )
        image_axes = tuple(
            (item.source, item.target, item.cover_path) for item in getattr(image, kind)
        )
        source_axes = tuple(
            (item.source, item.target, item.cover_path)
            for item in getattr(source, kind)
        )
        target_axes = tuple(
            (item.source, item.target, item.cover_path)
            for item in getattr(target, kind)
        )
        if (
            image_axes != expected_axes
            or source_axes != expected_axes
            or target_axes != expected_axes
        ):
            raise ValueError(
                f"image {kind} must retain the complete parent diagram axes"
            )
    expected_comparable_pairs = len(expected_cover_axes) + len(expected_derived_axes)
    expected_diamonds = 0
    for earlier in cells:
        for later in cells:
            if len(later) != len(earlier) + 2 or not set(earlier) < set(later):
                continue
            middles = sum(
                1
                for middle in cells
                if len(middle) == len(earlier) + 1
                and set(earlier) < set(middle) < set(later)
            )
            expected_diamonds += middles * (middles - 1) // 2
    for role, sheaf in (
        ("image", image),
        ("source", source),
        ("target", target),
    ):
        if (
            sheaf.diamonds != expected_diamonds
            or sheaf.comparable_pairs != expected_comparable_pairs
        ):
            raise ValueError(
                f"{role} must retain the diagram counters derived from its complex"
            )


def _require_component_shapes(
    components: tuple[Component, ...],
    cells: tuple[tuple[str, ...], ...],
    source: FiniteCellularSheaf,
    target: FiniteCellularSheaf,
) -> None:
    source_ranks = {stalk.simplex: len(stalk.basis) for stalk in source.stalks}
    target_ranks = {stalk.simplex: len(stalk.basis) for stalk in target.stalks}
    for cell, matrix in zip(cells, components, strict=True):
        raw = matrix[1]
        if len(raw) != target_ranks[cell] or any(
            len(row) != source_ranks[cell] for row in raw
        ):
            raise _domain(
                "component_shape", "morphism components must match their stalk axes"
            )


def _induced_restriction_work(
    item: SheafRestriction,
    source_ranks: dict[tuple[str, ...], int],
    target_ranks: dict[tuple[str, ...], int],
) -> int:
    """Bound restriction multiplication and the induced image-coordinate solve."""
    lower, upper = item.source, item.target
    lower_target_rank = target_ranks[lower]
    upper_target_rank = target_ranks[upper]
    lower_image_rank = min(source_ranks[lower], lower_target_rank)
    upper_image_rank = min(source_ranks[upper], upper_target_rank)
    multiply = upper_target_rank * lower_target_rank * lower_image_rank
    solve = 2 * upper_target_rank**2 * (lower_image_rank + upper_image_rank)
    return max(1, multiply) + max(1, solve)


def _require_component_axes(
    components: tuple[Component, ...],
    cells: tuple[tuple[str, ...], ...],
    row_ranks: dict[tuple[str, ...], int],
    column_ranks: dict[tuple[str, ...], int],
    label: str,
) -> None:
    if tuple(key for key, _matrix in components) != cells:
        raise ValueError(f"{label} components must retain canonical simplex axes")
    for cell, matrix in components:
        if not isinstance(cell, tuple):
            raise ValueError(f"{label} components need simplex tuple axes")
        if len(matrix) != row_ranks[cell] or any(
            len(row) != column_ranks[cell] for row in matrix
        ):
            raise ValueError(f"{label} matrices must match the stalk axes")


def _canonical_component_cells(
    components: tuple[Component, ...], cells: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(components, tuple) or len(components) != len(cells):
        raise _domain(
            "component_axis",
            "one morphism component per simplex in canonical order is required",
        )
    if any(
        not isinstance(component, tuple)
        or len(component) != 2
        or not isinstance(component[0], (str, tuple))
        or not isinstance(component[1], (tuple, list))
        or any(not isinstance(row, (tuple, list)) for row in component[1])
        for component in components
    ):
        raise _domain(
            "component_structure", "morphism components must be (simplex, matrix) pairs"
        )
    keys = tuple(_resolve_component_key(key, cells) for key, _matrix in components)
    if keys != cells:
        raise _domain(
            "component_axis",
            "one morphism component per simplex in canonical order is required",
        )
    return keys


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
    if type(value) is not SheafMorphismResult:
        raise _domain("morphism_type", "input must be a typed sheaf morphism")
    if (
        type(getattr(value, "source", None)) is not FiniteCellularSheaf
        or type(getattr(value, "target", None)) is not FiniteCellularSheaf
    ):
        raise _domain(
            "parent_type", "morphism parents must be canonical finite cellular sheaves"
        )
    source, target = value.source, value.target
    # Bound the parent container sizes before the recursive rebuild below. A
    # `model_construct` carrier can hold an arbitrarily large `stalks`, basis, or
    # restriction container, and `_unvalidated_payload` copies every nested
    # element before any rank, cell-count, work, or output admission runs, so
    # malformed input could force unbounded CPU and memory merely to reach
    # validation. Bound each container before walking its already-bounded children.
    _reject_oversized_parent_container(source, "source")
    _reject_oversized_parent_container(target, "target")
    try:
        source = FiniteCellularSheaf.model_validate(_unvalidated_payload(source))
        target = FiniteCellularSheaf.model_validate(_unvalidated_payload(target))
    except (ValidationError, AttributeError, TypeError, ValueError) as error:
        raise _domain(
            "parent_structure",
            "morphism parents must contain structurally valid cellular sheaf data",
        ) from error
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
    cells = source.canonical_face_order
    _canonical_component_cells(value.components, cells)
    _require_component_shapes(value.components, cells, source, target)
    target_cover, input_digits, morphism_work = _admit_morphism_resources(
        source, target, value.components
    )
    source_stalks = {stalk.simplex: stalk for stalk in source.stalks}
    target_stalks = {stalk.simplex: stalk for stalk in target.stalks}
    source_ranks = {cell: len(stalk.basis) for cell, stalk in source_stalks.items()}
    target_ranks = {cell: len(stalk.basis) for cell, stalk in target_stalks.items()}
    components: dict[tuple[str, ...], tuple[tuple[Scalar, ...], ...]] = {}
    canonical_components: list[Component] = []
    for cell, (_key, raw) in zip(cells, value.components, strict=True):
        matrix = _admit_component_matrix(raw, field, ("components", ".".join(cell)))
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
        work += _induced_restriction_work(item, source_ranks, target_ranks)
    reconstruction_work, reconstruction_digit_work = _parent_reconstruction_bounds(
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
        MAX_SHEAF_MORPHISM_PARENT_CELLS
        + reconstruction_digit_work
        + sheaf_scalar_digit_work(output_cells, output_digits)
        > MAX_SHEAF_MORPHISM_OUTPUT_DIGIT_WORK
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


__all__ = ["SheafMorphismImageResult", "image_of_morphism"]
