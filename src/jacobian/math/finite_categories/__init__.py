"""Exact bounded finite-category values and constructions."""

from jacobian.math.finite_categories.operations import (
    category_profile,
    opposite_category,
    product,
)
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
    "category_profile",
    "opposite_category",
    "product",
]
