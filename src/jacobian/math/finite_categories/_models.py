"""Typed wire contracts for finite category operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_OBJECTS = 20
MAX_MORPHISMS = 100


class MorphismSpec(StrictModel):
    """One morphism: source and target objects, plus a unique ID."""

    morphism_id: str
    source: str
    target: str


class FiniteCategoryRequest(StrictModel):
    """A finite category presented extensionally."""

    objects: tuple[str, ...] = Field(min_length=1, max_length=MAX_OBJECTS)
    morphisms: tuple[MorphismSpec, ...] = Field(max_length=MAX_MORPHISMS)

    @model_validator(mode="after")
    def require_valid_category(self) -> Self:
        obj_set = set(self.objects)
        if len(obj_set) != len(self.objects):
            raise ValueError("object labels must be distinct")
        morph_ids = [m.morphism_id for m in self.morphisms]
        if len(set(morph_ids)) != len(morph_ids):
            raise ValueError("morphism IDs must be distinct")
        for m in self.morphisms:
            if m.source not in obj_set or m.target not in obj_set:
                raise ValueError(
                    "every morphism source/target must be a declared object"
                )
        return self


class CategoryProfileResult(StrictModel):
    """Profile of a finite category: hom-sets, endomorphisms, isomorphisms."""

    objects: tuple[str, ...]
    num_objects: int
    num_morphisms: int
    hom_sets: tuple[tuple[str, int], ...]
    endomorphisms: tuple[tuple[str, int], ...]
    identity_morphisms: tuple[tuple[str, str], ...]


class OppositeCategoryResult(StrictModel):
    """The opposite category with reversed morphisms."""

    objects: tuple[str, ...]
    morphisms: tuple[MorphismSpec, ...]
