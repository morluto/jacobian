"""Exact bounded finite-category values and constructions."""

from jacobian.math.finite_categories.operations import product
from jacobian.math.finite_categories.values import (
    CategoryIdentifier,
    FiniteCategory,
    FiniteCategoryProduct,
    MorphismSpec,
    ProductMorphismProjection,
    ProductObjectProjection,
)

__all__ = [
    "CategoryIdentifier",
    "FiniteCategory",
    "FiniteCategoryProduct",
    "MorphismSpec",
    "ProductMorphismProjection",
    "ProductObjectProjection",
    "product",
]
