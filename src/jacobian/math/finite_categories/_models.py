"""Typed wire contracts for finite category operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel

MAX_OBJECTS = 20
MAX_MORPHISMS = 100
MAX_COMPOSITIONS = 1000


class MorphismSpec(StrictModel):
    """One morphism: source and target objects, plus a unique ID."""

    morphism_id: OpaqueLabel
    source: OpaqueLabel
    target: OpaqueLabel


def _morphism_index(
    objects: tuple[str, ...], morphisms: tuple[MorphismSpec, ...]
) -> dict[str, MorphismSpec]:
    obj_set = set(objects)
    if len(obj_set) != len(objects):
        raise ValueError("object labels must be distinct")
    by_id = {m.morphism_id: m for m in morphisms}
    if len(by_id) != len(morphisms):
        raise ValueError("morphism IDs must be distinct")
    for m in morphisms:
        if m.source not in obj_set or m.target not in obj_set:
            raise ValueError("every morphism source/target must be a declared object")
    return by_id


def _identity_map(
    objects: tuple[str, ...],
    by_id: dict[str, MorphismSpec],
    identities: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    obj_set = set(objects)
    id_map: dict[str, str] = {}
    for obj, morph_id in identities:
        if obj not in obj_set:
            raise ValueError("identity objects must be declared objects")
        if obj in id_map:
            raise ValueError("each object has exactly one identity")
        m = by_id.get(morph_id)
        if m is None:
            raise ValueError("an identity must name a declared morphism")
        if m.source != obj or m.target != obj:
            raise ValueError("an identity must be an endomorphism of its object")
        id_map[obj] = morph_id
    if set(id_map) != obj_set:
        raise ValueError("every object must have exactly one identity")
    return id_map


def _composition_table(
    morphisms: tuple[MorphismSpec, ...],
    by_id: dict[str, MorphismSpec],
    composition: tuple[tuple[str, str, str], ...],
) -> dict[tuple[str, str], str]:
    comp: dict[tuple[str, str], str] = {}
    for g, f, result in composition:
        if g not in by_id or f not in by_id or result not in by_id:
            raise ValueError("composition must name declared morphisms")
        ff = by_id[f]
        gf = by_id[g]
        rf = by_id[result]
        if ff.target != gf.source:
            raise ValueError("composition requires target(f) == source(g)")
        if rf.source != ff.source or rf.target != gf.target:
            raise ValueError("composition result must have source(f) and target(g)")
        if (g, f) in comp:
            raise ValueError("composition must be total and single-valued")
        comp[(g, f)] = result

    composable = {
        (g.morphism_id, f.morphism_id)
        for f in morphisms
        for g in morphisms
        if f.target == g.source
    }
    if set(comp) != composable:
        raise ValueError(
            "composition table domain must be exactly the composable pairs"
        )
    return comp


def _check_unit_laws(
    morphisms: tuple[MorphismSpec, ...],
    id_map: dict[str, str],
    comp: dict[tuple[str, str], str],
) -> None:
    for m in morphisms:
        id_target = id_map[m.target]
        id_source = id_map[m.source]
        if comp[(id_target, m.morphism_id)] != m.morphism_id:
            raise ValueError("left identity law violated")
        if comp[(m.morphism_id, id_source)] != m.morphism_id:
            raise ValueError("right identity law violated")


def _check_associativity(
    morphisms: tuple[MorphismSpec, ...], comp: dict[tuple[str, str], str]
) -> None:
    for h in morphisms:
        for g in morphisms:
            for f in morphisms:
                if f.target == g.source and g.target == h.source:
                    hg = comp[(h.morphism_id, g.morphism_id)]
                    gf = comp[(g.morphism_id, f.morphism_id)]
                    if comp[(hg, f.morphism_id)] != comp[(h.morphism_id, gf)]:
                        raise ValueError("associativity violated")


def _check_category(
    objects: tuple[str, ...],
    morphisms: tuple[MorphismSpec, ...],
    identities: tuple[tuple[str, str], ...],
    composition: tuple[tuple[str, str, str], ...],
) -> None:
    """Validate the defining category data and laws, raising on violation.

    A finite category is presented extensionally by its objects, morphisms,
    a designated identity per object, and a total composition table.  This
    enforces that the input is a genuine category: distinct labels, closed
    source/target domains, exactly one typed identity per object, a
    composition table whose domain is exactly the composable pairs, correct
    result typing, both unit laws, and associativity on every composable
    triple.  ``composition`` entries are ``(g, f, result)`` with ``g . f``.
    """

    by_id = _morphism_index(objects, morphisms)
    id_map = _identity_map(objects, by_id, identities)
    comp = _composition_table(morphisms, by_id, composition)
    _check_unit_laws(morphisms, id_map, comp)
    _check_associativity(morphisms, comp)


class FiniteCategoryRequest(StrictModel):
    """A finite category presented extensionally.

    ``identities`` pairs each object with its designated identity morphism;
    ``composition`` carries a total result for every composable pair as
    ``(g, f, result)`` meaning ``g . f = result``.  The complete category
    laws are enforced once at the value boundary.
    """

    objects: tuple[OpaqueLabel, ...] = Field(min_length=1, max_length=MAX_OBJECTS)
    morphisms: tuple[MorphismSpec, ...] = Field(max_length=MAX_MORPHISMS)
    identities: tuple[tuple[OpaqueLabel, OpaqueLabel], ...] = Field(
        max_length=MAX_OBJECTS
    )
    composition: tuple[tuple[OpaqueLabel, OpaqueLabel, OpaqueLabel], ...] = Field(
        max_length=MAX_COMPOSITIONS
    )

    @model_validator(mode="after")
    def require_valid_category(self) -> Self:
        _check_category(self.objects, self.morphisms, self.identities, self.composition)
        return self


class CategoryProfileResult(StrictModel):
    """Profile of a finite category: hom-sets, endomorphisms, identities."""

    objects: tuple[OpaqueLabel, ...]
    num_objects: int
    num_morphisms: int
    hom_sets: tuple[tuple[OpaqueLabel, OpaqueLabel, int], ...]
    endomorphisms: tuple[tuple[OpaqueLabel, int], ...]
    identity_morphisms: tuple[tuple[OpaqueLabel, OpaqueLabel], ...]


class OppositeCategoryResult(StrictModel):
    """The opposite category: reversed morphisms and reversed composition."""

    objects: tuple[OpaqueLabel, ...]
    morphisms: tuple[MorphismSpec, ...]
    identities: tuple[tuple[OpaqueLabel, OpaqueLabel], ...]
    composition: tuple[tuple[OpaqueLabel, OpaqueLabel, OpaqueLabel], ...]

    @model_validator(mode="after")
    def require_valid_opposite(self) -> Self:
        _check_category(self.objects, self.morphisms, self.identities, self.composition)
        return self
