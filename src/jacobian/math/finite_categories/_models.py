"""Typed wire contracts for finite category operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.finite_categories._product import (
    CategoryProductAdmissionError,
    _product_error,
    _product_plan,
)
from jacobian.math.finite_categories.values import (
    MAX_CATEGORY_MORPHISMS,
    MAX_CATEGORY_OBJECTS,
    CategoryIdentifier,
    FiniteCategory,
    MorphismSpec,
)


class CategoryProfileResult(StrictModel):
    """A bounded profile claim for one finite category.

    Its source and result shape are structural.  The owner-local verifier
    checks the derived hom-set and endomorphism counts for supplied claims.
    """

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

        return cls(
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

    @model_validator(mode="after")
    def require_bounded_product(self) -> Self:
        try:
            _product_plan(self.left, self.right)
        except CategoryProductAdmissionError as exc:
            raise _product_error(exc.reason, str(exc)) from None
        return self


__all__ = [
    "CategoryProductRequest",
    "CategoryProfileResult",
    "MorphismSpec",
]
