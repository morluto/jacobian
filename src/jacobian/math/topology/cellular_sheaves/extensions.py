"""Sections, restriction queries, and natural morphisms of finite sheaves."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
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


class SheafMorphismRequest(StrictModel):
    source: FiniteCellularSheaf
    target: FiniteCellularSheaf
    components: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = Field(default=())


class SheafMorphismResult(StrictModel):
    source: FiniteCellularSheaf
    target: FiniteCellularSheaf
    components: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]
    natural: bool
    obstruction: str | None = None


def sections(sheaf: FiniteCellularSheaf) -> SheafSectionsResult:
    cohomology = sheaf_cohomology(sheaf)
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


def _mul(a: Any, b: Any, p: int | None) -> Any:
    if not a or not b:
        return [[] for _ in a]
    cols = list(zip(*b, strict=False))
    return [
        [sum(x * y for x, y in zip(row, col, strict=False)) for col in cols]
        for row in a
    ]


def _transpose(a: Any) -> Any:
    return list(map(list, zip(*a, strict=False))) if a else []


def morphism(
    source: FiniteCellularSheaf,
    target: FiniteCellularSheaf,
    components: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...],
) -> SheafMorphismResult:
    _complete_diagram(source, role="source")
    _complete_diagram(target, role="target")
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
    stalk = {".".join(s.simplex): s for s in source.stalks}
    target_stalk = {".".join(s.simplex): s for s in target.stalks}
    given = dict(components)
    expected = tuple(".".join(s.simplex) for s in source.stalks)
    if tuple(key for key, _ in components) != expected:
        raise OperationDomainValidationError(
            location=("components",),
            code="cellular_sheaf.morphism_component_axis",
            message="one component in canonical stalk order is required",
        )
    p = source.prime
    for key, matrix in components:
        rows = len(target_stalk[key].basis)
        cols = len(stalk[key].basis)
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            raise OperationDomainValidationError(
                location=("components", key),
                code="cellular_sheaf.morphism_shape",
                message="component axes do not match stalk ranks",
            )
    target_cover = {
        (item.source, item.target): item for item in target.cover_restrictions
    }
    try:
        for restriction in source.cover_restrictions:
            a = ".".join(restriction.source)
            b = ".".join(restriction.target)
            left = _mul(
                [[_mat(x, p) for x in row] for row in given[b]],
                [[_mat(x, p) for x in row] for row in restriction.entries],
                p,
            )
            tr = target_cover[(restriction.source, restriction.target)]
            right = _mul(
                [[_mat(x, p) for x in row] for row in tr.entries],
                [[_mat(x, p) for x in row] for row in given[a]],
                p,
            )
            if left != right:
                return SheafMorphismResult(
                    source=source,
                    target=target,
                    components=components,
                    natural=False,
                    obstruction=f"naturality fails on {a} < {b}",
                )
    except (KeyError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise OperationDomainValidationError(
            location=("components",),
            code="cellular_sheaf.morphism_scalar_invalid",
            message="sheaf components and restrictions must contain valid exact scalars",
        ) from error
    return SheafMorphismResult(
        source=source, target=target, components=components, natural=True
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
