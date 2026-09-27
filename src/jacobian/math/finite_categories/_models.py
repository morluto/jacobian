"""Typed wire contracts for finite category operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.finite_categories.values import (
    MAX_CATEGORY_MORPHISMS,
    MAX_CATEGORY_OBJECTS,
    CategoryIdentifier,
    FiniteCategory,
    MorphismSpec,
)
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet


class CategoryProfileResult(StrictModel):
    """A bounded hom-set profile for one finite category.

    Its source and result shape are structural. Defining count evidence belongs
    to the producing kernel's tests rather than result deserialization.
    """

    category: FiniteCategory
    objects: tuple[CategoryIdentifier, ...] = Field(max_length=MAX_CATEGORY_OBJECTS)
    num_objects: int = Field(ge=0, le=MAX_CATEGORY_OBJECTS)
    num_morphisms: int = Field(ge=0, le=MAX_CATEGORY_MORPHISMS)
    hom_sets: tuple[tuple[CategoryIdentifier, CategoryIdentifier, int], ...] = Field(
        max_length=MAX_CATEGORY_MORPHISMS
    )
    endomorphisms: tuple[tuple[CategoryIdentifier, int], ...] = Field(
        max_length=MAX_CATEGORY_OBJECTS
    )
    identity_morphisms: tuple[tuple[CategoryIdentifier, CategoryIdentifier], ...] = (
        Field(max_length=MAX_CATEGORY_OBJECTS)
    )

    @classmethod
    def _from_kernel(
        cls,
        category: FiniteCategory,
        hom_sets: tuple[tuple[CategoryIdentifier, CategoryIdentifier, int], ...],
        endomorphisms: tuple[tuple[CategoryIdentifier, int], ...],
    ) -> Self:
        """Build a profile result from the trusted owner-local kernel."""

        return cls.model_construct(
            category=category,
            objects=category.objects,
            num_objects=len(category.objects),
            num_morphisms=len(category.morphisms),
            hom_sets=hom_sets,
            endomorphisms=endomorphisms,
            identity_morphisms=category.identities,
        )


class CategoryProductRequest(StrictModel):
    """Two canonical finite categories whose Cartesian product is requested."""

    left: FiniteCategory
    right: FiniteCategory


class CategoryNerveRequest(StrictModel):
    """A finite category and requested degree of its nerve prefix."""

    category: FiniteCategory
    max_degree: int = Field(ge=0, le=4)


class FiniteCategoryNerve(StrictModel):
    """A nerve prefix with its exact source category and simplex transport.

    ``simplex_morphisms[n][j]`` records the composable morphism tuple for the
    simplex labelled ``simplicial_set.sets[n][j]``.  In degree zero this is
    the empty tuple; ``simplex_objects`` records its vertex.
    """

    category: FiniteCategory
    simplicial_set: FiniteTruncatedSimplicialSet
    simplex_morphisms: tuple[tuple[tuple[CategoryIdentifier, ...], ...], ...]
    simplex_objects: tuple[tuple[tuple[CategoryIdentifier, ...], ...], ...]

    @model_validator(mode="after")
    def require_aligned_simplex_transport(self) -> Self:
        if not (
            len(self.simplex_morphisms)
            == len(self.simplex_objects)
            == self.simplicial_set.max_degree + 1
        ):
            raise ValueError("nerve transport must cover every simplicial degree")
        for degree, (morphism_level, object_level, labels) in enumerate(
            zip(
                self.simplex_morphisms,
                self.simplex_objects,
                self.simplicial_set.sets,
                strict=True,
            )
        ):
            if not (len(morphism_level) == len(object_level) == len(labels)):
                raise ValueError(
                    "nerve transport must align with canonical simplex labels"
                )
            if any(
                len(morphisms) != degree or len(objects) != degree + 1
                for morphisms, objects in zip(morphism_level, object_level, strict=True)
            ):
                raise ValueError(
                    "nerve simplex transport has the wrong simplex dimension"
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        category: FiniteCategory,
        simplicial_set: FiniteTruncatedSimplicialSet,
        simplex_morphisms: tuple[tuple[tuple[CategoryIdentifier, ...], ...], ...],
        simplex_objects: tuple[tuple[tuple[CategoryIdentifier, ...], ...], ...],
    ) -> Self:
        return cls.model_construct(
            category=category,
            simplicial_set=simplicial_set,
            simplex_morphisms=simplex_morphisms,
            simplex_objects=simplex_objects,
        )


__all__ = [
    "CategoryNerveRequest",
    "CategoryProductRequest",
    "CategoryProfileResult",
    "FiniteCategoryNerve",
    "MorphismSpec",
]
