"""Exact bounded finite-category values and constructions."""

from jacobian.math.finite_categories._models import (
    CategoryNerveRequest,
    FiniteCategoryNerve,
)
from jacobian.math.finite_categories.nerve import nerve_prefix
from jacobian.math.finite_categories.operations import (
    category_profile,
    opposite_category,
    product,
    verify_category_profile,
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
    "CategoryNerveRequest",
    "FiniteCategory",
    "FiniteCategoryNerve",
    "FiniteCategoryProduct",
    "MorphismSpec",
    "ProductMorphismProjection",
    "ProductObjectProjection",
    "category_profile",
    "nerve_prefix",
    "opposite_category",
    "product",
    "verify_category_profile",
]
