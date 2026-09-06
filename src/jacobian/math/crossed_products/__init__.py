"""Exact finite-coset crossed products over prime fields."""

from jacobian.math.crossed_products.operations import multiply, verify_multiply
from jacobian.math.crossed_products.values import (
    FiniteCosetCrossedProductElement,
    FiniteCosetCrossedProductPresentation,
    FiniteCosetCrossedProductTerm,
)

__all__ = [
    "FiniteCosetCrossedProductElement",
    "FiniteCosetCrossedProductPresentation",
    "FiniteCosetCrossedProductTerm",
    "multiply",
    "verify_multiply",
]
