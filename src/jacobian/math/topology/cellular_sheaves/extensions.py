"""Sections, restriction queries, and natural morphisms of finite sheaves."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.cellular_sheaves._kernel import _admit_field
from jacobian.math.topology.cellular_sheaves._models import (
    FiniteCellularSheaf,
    SheafRestriction,
)
from jacobian.math.topology.cellular_sheaves.operations import sheaf_cohomology


class SheafSectionsRequest(StrictModel):
    sheaf: FiniteCellularSheaf


class SheafSectionsResult(StrictModel):
    sheaf: FiniteCellularSheaf
    dimension: int
    basis_coordinates: tuple[tuple[str, ...], ...]
    cochain_dimension: int


class SheafRestrictionRequest(StrictModel):
    sheaf: FiniteCellularSheaf
    source: tuple[str, ...]
    target: tuple[str, ...]


class SheafRestrictionResult(StrictModel):
    sheaf: FiniteCellularSheaf
    restriction: SheafRestriction


ComponentKey = str | tuple[str, ...]
Component = tuple[ComponentKey, tuple[tuple[str, ...], ...]]


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


def sections(sheaf: FiniteCellularSheaf) -> SheafSectionsResult:
    cohomology = sheaf_cohomology(sheaf)
    # The empty complex has no cochain degrees. Its global section space is
    # therefore the zero vector space, rather than a missing degree-zero group.
    if not cohomology.groups:
        return SheafSectionsResult(
            sheaf=sheaf,
            dimension=0,
            basis_coordinates=(),
            cochain_dimension=0,
        )
    group = cohomology.groups[0]
    return SheafSectionsResult(
        sheaf=sheaf,
        dimension=group.betti_number,
        basis_coordinates=group.cocycle_representatives,
        cochain_dimension=group.cochain_dimension,
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


def _mat(entry: str, prime: int | None) -> Any:
    if prime is not None:
        return int(entry) % prime
    if "/" in entry:
        a, b = entry.split("/", 1)
        from fractions import Fraction

        return Fraction(int(a), int(b))
    from fractions import Fraction

    return Fraction(int(entry))


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
    target_cover = {
        (item.source, item.target): item for item in target.cover_restrictions
    }
    try:
        for restriction in source.cover_restrictions:
            a = restriction.source
            b = restriction.target
            left = _mul(
                [list(row) for row in given[b]],
                [[_mat(x, p) for x in row] for row in restriction.entries],
                p,
                output_width=len(stalk[a].basis),
            )
            tr = target_cover[(restriction.source, restriction.target)]
            right = _mul(
                [[_mat(x, p) for x in row] for row in tr.entries],
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


__all__ = [
    "SheafMorphismRequest",
    "SheafMorphismResult",
    "SheafRestrictionRequest",
    "SheafRestrictionResult",
    "SheafSectionsRequest",
    "SheafSectionsResult",
    "morphism",
    "restriction",
    "sections",
]
