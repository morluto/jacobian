"""Sections, restriction queries, and natural morphisms of finite sheaves."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    FiniteCellularSheaf,
    SheafOutcome,
    SheafRestriction,
)
from jacobian.math.topology.cellular_sheaves.operations import (
    from_cover_maps,
    sheaf_cohomology,
)


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


def _establish_complete_sheaf(
    sheaf: FiniteCellularSheaf, location: str
) -> FiniteCellularSheaf:
    # Serialized FiniteCellularSheaf values retain structural axes but not
    # trusted provenance. Re-establish the complete diagram at this consumer.
    cover_maps = tuple(
        CoverRestrictionMatrix(
            source=item.source, target=item.target, entries=item.entries
        )
        for item in sheaf.cover_restrictions
    )
    result = from_cover_maps(
        sheaf.complex,
        sheaf.coefficient_field,
        sheaf.prime,
        sheaf.stalks,
        cover_maps,
    )
    if result.outcome is not SheafOutcome.CELLULAR_SHEAF or result.sheaf is None:
        code = (
            result.obstruction.code.value.lower()
            if result.obstruction is not None
            else "invalid_diagram"
        )
        raise OperationDomainValidationError(
            location=(location,),
            code=f"cellular_sheaf.morphism.{code}",
            message=(
                result.obstruction.message
                if result.obstruction is not None
                else "the sheaf does not establish a complete restriction diagram"
            ),
        )
    return result.sheaf


def morphism(
    source: FiniteCellularSheaf,
    target: FiniteCellularSheaf,
    components: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...],
) -> SheafMorphismResult:
    source = _establish_complete_sheaf(source, "source")
    target = _establish_complete_sheaf(target, "target")
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
    for restriction in source.cover_restrictions:
        a = ".".join(restriction.source)
        b = ".".join(restriction.target)
        try:
            left = _mul(
                [[_mat(x, p) for x in row] for row in given[b]],
                [[_mat(x, p) for x in row] for row in restriction.entries],
                p,
            )
            tr = next(
                x
                for x in target.cover_restrictions
                if x.source == restriction.source and x.target == restriction.target
            )
            right = _mul(
                [[_mat(x, p) for x in row] for row in tr.entries],
                [[_mat(x, p) for x in row] for row in given[a]],
                p,
            )
        except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError) as exc:
            raise OperationDomainValidationError(
                location=("components", key),
                code="cellular_sheaf.morphism.entry_invalid",
                message="components and restriction scalars must use the declared exact field",
            ) from exc
        if left != right:
            return SheafMorphismResult(
                source=source,
                target=target,
                components=components,
                natural=False,
                obstruction=f"naturality fails on {a} < {b}",
            )
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
