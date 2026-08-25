"""Typed wire contracts for finite category operations."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.finite_categories.operations import (
    CategoryProductAdmissionError,
    _product_error,
    _product_plan,
)
from jacobian.math.finite_categories.values import (
    CategoryIdentifier,
    FiniteCategory,
    MorphismSpec,
)


class CategoryProfileResult(StrictModel):
    """Profile of a finite category: hom-sets, endomorphisms, identities."""

    objects: tuple[CategoryIdentifier, ...]
    num_objects: int
    num_morphisms: int
    hom_sets: tuple[tuple[CategoryIdentifier, CategoryIdentifier, int], ...]
    endomorphisms: tuple[tuple[CategoryIdentifier, int], ...]
    identity_morphisms: tuple[tuple[CategoryIdentifier, CategoryIdentifier], ...]


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
